Personal Repo for Simple Image Editing based on Denoising Diffusion Model

Reference:
https://github.com/openai/guided-diffusion


## Some results:

```
Column : sigma = arange(start=0.5, end=4, step=0.5)

Row : Edit-image + Diffused-image + Recovered-image
```

### cat edit


<img src="image_try/cat_1.png" alt="drawing" width="200"/> <img src="image_try/cat_2.png" alt="drawing" width="200"/>

CAT-(1)
![](image_try/cat_2/sample_-1/diff_sigma.jpg)
CAT_(2)
![](image_try/cat_1/sample_-1/diff_sigma.jpg)

### tv-edit

<img src="image_try/tv.png" alt="drawing" width="200"/>

TV-(1)

![](image_try/tv/sample_-1/diff_sigma.jpg)

### horse-edit

<img src="image_try/horse_1.png" alt="drawing" width="200"/>

HORSE-(1)

![](image_try/horse_1/sample_-1/diff_sigma.jpg)

