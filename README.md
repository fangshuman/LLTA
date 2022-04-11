# Learning to Learn Transferable Attack

This is the project page of our paper:

**Learning to Learn Transferable Attack**  
Fang, S., Li, J., Lin, X., & Ji, R.  
AAAI 2022. [arxiv](https://arxiv.org/abs/2112.06658)


## Run

+ `images/dev_dataset.csv` contains only URLs of the images. Download these images and put them into `images`
+ Generate adverial examples and the results will be saved in `output`
  + ResNet-50 as the source model: `python main.py --source-model resnet50`
  + DenseNet-121 as the source model: `python main.py --source-model densenet121` 
+ Download the adversarially trained models in [here](https://github.com/JHL-HUST/SI-NI-FGSM)
+ To evaluate success rate
  + Attack naturally trained models: `python eval/evaluate_NT_trained.py --adv-dir output` 
  + Attack adversarially trained models: `python eval/evaluate_AT_trained.py --adv-dir output` 
  