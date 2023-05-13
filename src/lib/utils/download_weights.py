import gdown

url = 'https://drive.google.com/uc?id=1wSoA5fm_d6JBZk4RZ1SzWLMgev4WqH21'
output = 'src/lib/asyrp/pretrained/celeba_hq.ckpt'
gdown.download(url, output, quiet=False)

url = 'https://drive.google.com/uc?id=1rJO1QSm1Rqp6JmHHNKcPni6maqbgKj0R'
output = 'src/lib/asyrp/pretrained/celebahq_pt2.pt'
gdown.download(url, output, quiet=False)

