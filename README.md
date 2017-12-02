
# Semi-supervised-GAN
there is the Tensorflow implement of 'Improved Techniques for Training GANs '
# Descriptioins
This project is a Tensorflow implemention of semi-supervised which described in [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498). We made some changes without changing the original intention. We train our code in mnist dataset, and achieve 0.99 test accuracy only with 200 labeled images. The test acc process is shown followed.
![test_acc](https://github.com/LDOUBLEV/semi-supervised-GAN/blob/master/plot/test_acc.jpg)
# usage 
The Tensorflow vision is 1.2 and we use python 2.7 to implement our code.
## train 
`python main.py --model=DCGAN --trainable=True --load_model=False 
 `
## test 
`python main.py --model=DCGAN  --trainable=False  --load_model=True
'


                `
