import gdown

url = 'https://drive.google.com/uc?id=1wSoA5fm_d6JBZk4RZ1SzWLMgev4WqH21'
output = 'src/lib/Asyrp_official/pretrained/celeba_hq.ckpt'
gdown.download(url, output, quiet=False)