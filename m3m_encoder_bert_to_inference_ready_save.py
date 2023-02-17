import torch
from app.config import m3m  as m3m_config
from app.kgqa.m3m import EncoderBERT


if __name__ == '__main__':
    encoder = torch.load("/home/salnikov/data/m3m/ckpts/encoder", map_location='cpu')
    torch.save(encoder.state_dict(), '/home/salnikov/data/m3m/ckpts/encoder.pt')
