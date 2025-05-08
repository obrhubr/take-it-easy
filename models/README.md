# Training

When I first looked at the repository [polarbart/TakeItEasyAI](https://github.com/polarbart/TakeItEasyAI) and tried to reproduce the results, I wished that I had more details about his attempts and different experiments.

To make life simpler for those also wanting to train their own model, I have included `training_runs.json` and `full_results.json`. This contains a full log of all experiments and their results.

The first file (`training_runs.json`) contains an overview of all hyperparameter configurations I tried and the best results I got. The second file (`full_results.json`) contains the results for each iteration of the training runs, with which you can reconstruct the progress during training.

If you need any more data or have ideas about configurations that could improve upon the existings ones, please contact me @ the mail address listed on my [website](https://obrhubr.org).

## An overview of training results in graphs:

```
Name: default - mean=168.06, min=mean=55.0, max=mean=255.0
Name: hidden-4096 - mean=166.54, min=mean=48.0, max=mean=267.0
Name: output-150 - mean=166.27, min=mean=65.0, max=mean=258.0
Name: lr-0.001-eps-0.6 - mean=166.09, min=mean=45.0, max=mean=249.0
Name: lr-0.001 - mean=166.85, min=mean=60.0, max=mean=275.0
Name: eps-0.6 - mean=160.81, min=mean=52.0, max=mean=267.0
Name: eps-0.8 - mean=164.08, min=mean=55.0, max=mean=258.0
Name: lr-0.001-eps-0.5 - mean=167.35, min=mean=45.0, max=mean=269.0
Name: hidden-4092-lr-0.001-eps-0.6 - mean=166.54, min=mean=57.0, max=mean=255.0
Name: hidden-512-lr-0.001-eps-0.6 - mean=164.69, min=mean=52.0, max=mean=268.0
Name: hidden-1024-lr-0.001-eps-0.6 - mean=166.48, min=mean=54.0, max=mean=260.0
Name: hidden-1024-lr-0.01-eps-0.6 - mean=160.89, min=mean=37.0, max=mean=267.0
```

All of the training runs (no matter the parameters) resulted in models that perform nearly equally well (a mean score of around ~167). The differences (as you'll see on the graphs below) is in how quickly they managed to get to that score.

The mean score (and especially min and max) vary quite a bit, even over a sample of 64k games, so most of the final iterations will contain some random noise which might look like improvements. There was however an actual gradual improvement from about `166.5` to `168.05` over iterations 80-150 when I trained the default config (See the `full_results.json` file).

There are some slight differences in the lowest and highest scores achieved, but this could be due to statistical variations in the games played.

![Boxplots comparing the different results.](../.github/boxplots.png)

Some configurations train faster, especially if you bump up the learning rate and increase the number of games in the training corpus (from 16k to 32k or 64k).

But if the learning rate is too high or doesn't decay enough, the model will not be able to finetune well enough to achieve the final few points.

![Comparing scores during training.](../.github/scores.png)

The learning rate maxes out at the same time that scores over ~160 are achieved, which seems to be when the models begin to have difficulty getting any further.

![Comparing loss during training.](../.github/loss.png)

This one vs. one comparison of all models shows only slight differences between them, but there are some variations in mean score and the distribution of the scores.

![A one-to-one comparison of the different model's histograms](../.github/histograms.png)