# RE_U-NET
 This repository is a modified U-NET. Based on Pytorch.</br>
 For learning and operating on small scale dataset.

### Before Start：
This is my first and probably last academic practicum. I hope you enjoy it.
I coded this repository in my early days as a rookie programmer, and it has been with me for a long time. helped me deal with many difficult tasks. Now I'm improving it and making it open-source for all newcomers. I believe it could put a perfect end to my academic career.

## Important Issue
* this repository was directly built on latest Google Colab GPU Environment(torch 1.13.1+cu116)</br>
  Personally, I strongly recommend beginners to register a Google account.</br>
  Colab & Google drive can release you from environment setting.
* If you choose to run this repository on Colab.</br>Please read the <font color=orange>re_U_NET_tutorial_Colab.ipynb</font></br>
  This file includes the model structure and basic information about file tree.
* This a tutorial for my research group, and for those who want to start learning basic information about CV and Pytorch.</br></br>
* <font color=red>This repository is NOT a long-term support repository</font></br></br>

* <font color=orange>I did not and will not offer any Architecture Diagram, for reasons below:</font></br>
   1. This model is based on U-NET, which is a very classic model.
   2. Concluding the Architecture is a good training for rookies.
  3. ~~Tired of drawing Architecture Diagram~~
## Code Catalogue & Abstract
* <font color=orange>re_NET_lw.py:</font> Including structure of RE_U-NET
* <font color=orange>dataset.py:</font> Generate data loader
* <font color=orange>train.py:</font> model training
* <font color=orange>predict.py:</font> predicting to generate saved image(demo)

## Special Thanks
[Aladdin Persson](https://www.youtube.com/@AladdinPersson)</br>
His [video](https://www.youtube.com/watch?v=IHq1t7NxS8k&t=2s) about how to rebuild a original U-NET inspire me a lot.</br>
Institute: [Urban Mobility Institute of Tongji Uni](https://umi.tongji.edu.cn/index.htm)




