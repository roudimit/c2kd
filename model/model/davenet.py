import math

import torch
import torch.nn as nn
import librosa
import numpy as np
import scipy.signal


def conv1d(in_planes, out_planes, width=9, stride=1, bias=False, amd=False):
    """1xd convolution with padding"""
    if width % 2 == 0:
        pad_amt = int(width / 2)
    else:
        pad_amt = int((width - 1) / 2)
    if amd:
        return nn.Conv2d(in_planes, out_planes, kernel_size=(1, width), stride=stride, bias=bias)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=(1, width), stride=stride, padding=(0,pad_amt), bias=bias)


class SpeechBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, width=9, stride=1, downsample=None, amd=False):
        super(SpeechBasicBlock, self).__init__()
        self.amd = amd
        self.conv1 = conv1d(inplanes, planes, width=width, stride=stride, amd=amd)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1d(planes, planes, width=width, amd=amd)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        if self.amd:
            out += residual[:, :, :, :out.shape[3]]
        else:
            out += residual
        out = self.relu(out)
        return out


class ResDavenet(nn.Module):
    def __init__(self, feat_dim=40, block=SpeechBasicBlock, layers=[2, 2, 2, 2], layer_widths=[128, 128, 256, 512, 1024], convsize=9, amd=False):
        super(ResDavenet, self).__init__()
        self.feat_dim = feat_dim
        self.inplanes = layer_widths[0]
        self.amd = amd
        self.batchnorm1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=(self.feat_dim,1), stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, layer_widths[1], layers[0], width=convsize, stride=2)
        self.layer2 = self._make_layer(block, layer_widths[2], layers[1], width=convsize, stride=2)
        self.layer3 = self._make_layer(block, layer_widths[3], layers[2], width=convsize, stride=2)
        self.layer4 = self._make_layer(block, layer_widths[4], layers[3], width=convsize, stride=1 if self.amd else 2)
        if len(layers) == 6:
            self.layer5 = self._make_layer(block, layer_widths[5], layers[4], width=convsize, stride=2)
            self.layer6 = self._make_layer(block, layer_widths[6], layers[5], width=convsize, stride=2)
        else:
            self.layer5 = None
            self.layer6 = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, width=9, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, width=width, stride=stride, downsample=downsample, amd=self.amd))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, width=width, stride=1, amd=self.amd))
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.layer5 is not None:
            x = self.layer5(x)
            x = self.layer6(x)

        x = x.squeeze(2)
        return x

def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.
    
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """    
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

def load_DAVEnet(amd=False, v2=False):
    if v2:
        audio_model = ResDavenet(feat_dim=40, layers=[2,2,1,1,1,1], convsize=9,
                                 layer_widths=[128,128,256,512,1024,2048,4096], amd=amd)
    else:
        audio_model = ResDavenet(feat_dim=40, layers=[2, 2, 2, 2], convsize=9,
                                 layer_widths=[128, 128, 256, 512, 1024], amd=amd)

    return audio_model

def LoadAudio(path, target_length=2048, use_raw_length=False):
    audio_type = 'melspectrogram'
    preemph_coef = 0.97
    sample_rate = 16000
    window_size = 0.025
    window_stride = 0.01
    window_type = 'hamming'
    num_mel_bins = 40
    padval = 0
    fmin = 20
    n_fft = int(sample_rate * window_size)
    win_length = int(sample_rate * window_size)
    hop_length = int(sample_rate * window_stride)

    windows = {'hamming': scipy.signal.hamming}
    # load audio, subtract DC, preemphasis
    # sr=None to avoid resampling (assuming audio already at 16 kHz sr)
    y, sr = librosa.load(path, sr=None)
    if y.size == 0:
        y = np.zeros(200)
    y = y - y.mean()
    y = preemphasis(y, preemph_coef)
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length,
        window=windows[window_type])
    spec = np.abs(stft)**2
    if audio_type == 'melspectrogram':
        mel_basis = librosa.filters.mel(sr, n_fft, n_mels=num_mel_bins, fmin=fmin)
        melspec = np.dot(mel_basis, spec)
        feats = librosa.power_to_db(melspec, ref=np.max)
    n_frames = feats.shape[1]

    if use_raw_length:
        target_length = n_frames
    p = target_length - n_frames
    if p > 0:
        feats = np.pad(feats, ((0,0),(0,p)), 'constant',
            constant_values=(padval,padval))
    elif p < 0:
        feats = feats[:,0:p]
        n_frames = target_length

    return feats, n_frames