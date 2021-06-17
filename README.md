# Project for the AICS course (MLT LT2318)

This is a repository contains a set file structure for submitting your project in this course.

See `library/github-instructions.md` for a description how to use this repository.

Use this file to keep general notes about your project, including your individual project plan. We will also use it for comments.

## Project Description  
#### Main Aim
- __Research Question:__  How will a standard image captioning model perform in generating captions for the visually impaired?
- Implement the neural image caption generator with visual attention presented by [Xu et al., 2015](https://arxiv.org/abs/1502.03044?source=post_page---------------------------) and examine how well the model performs when tested on the [VizWiz Dataset](https://vizwiz.org/)

#### Model Architecture
- encoder-decoder with attention mechanism
- pretrained encoder (ResNet)
- greedy search for generating captions
- early stopping after 20 epochs of no improvement

#### Acknowledgments
- PyTorch Tutorial to Image Captioning by [Sagar Vinodababu](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning#training)
- [VizWiz Dataset](https://arxiv.org/abs/2002.08565)

#### References
- Anderson, P., He, X., Buehler, C., Teney, D., Johnson, M., Gould, S., & Zhang, L. (2018). [Bottom-up and top-down attention for image captioning and visual question answering](https://arxiv.org/abs/1707.07998). In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 6077-6086).
- Bahdanau, D., Cho, K., & Bengio, Y. (2014). [Neural machine translation by jointly learning to align and translate](https://arxiv.org/abs/1409.0473). arXiv preprint arXiv:1409.0473.
- Gurari, D., Zhao, Y., Zhang, M., & Bhattacharya, N. (2020, August). [Captioning images taken by people who are blind](https://arxiv.org/abs/2002.08565). In European Conference on Computer Vision (pp. 417-434). Springer, Cham.
- Lin, T. Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., ... & Zitnick, C. L. (2014, September). [Microsoft coco: Common objects in context. In European conference on computer vision (pp. 740-755)](https://arxiv.org/abs/1405.0312). Springer, Cham.
- Vinyals, O., Toshev, A., Bengio, S., & Erhan, D. (2015). [Show and tell: A neural image caption generator](https://arxiv.org/abs/1411.4555). In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3156-3164).
- Xu, K., Ba, J., Kiros, R., Cho, K., Courville, A., Salakhudinov, R., ... & Bengio, Y. (2015, June). [Show, attend and tell: Neural image caption generation with visual attention](https://arxiv.org/abs/1502.03044?source=post_page---------------------------). In International conference on machine learning (pp. 2048-2057). PMLR.