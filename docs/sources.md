# Dataset sources and rights notes

## Dataset used by the local experiment

The pipeline reads [`Tinsae/Ethiopian-foods`](https://huggingface.co/datasets/Tinsae/Ethiopian-foods). The dataset card lists 11 labels—beyaynetu, chechebsa, doro wat, fir-fir, genfo, kikil, kitfo, shekla tibs, shiro wat, tihlo, and tire siga—and describes the photos as collected from social media. The Hub viewer reports 1,097 rows and about 272 MB. Its card does **not** declare a license or identify individual photo rights holders.

This is the source used for the local experiment because it is publicly downloadable. Public availability is not permission to republish. Images and derived checkpoints remain local and are excluded from version control. This should not be treated as legal clearance for commercial use or model-weight distribution.

## Similar Roboflow listing (not assumed to be the same copy)

Roboflow Universe has a similarly named [Ethiopian food classification dataset](https://universe.roboflow.com/classification-gphzo/ethiopian-food) with the same 11 class names, 1,092 listed images, and a displayed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license. Its creator is shown as “classification”; a dataset description is not published. Roboflow's [official export instructions](https://docs.roboflow.com/datasets/versions/dataset-versions/exporting-data) explain that exporting a version returns an account-key-protected download link. This environment did not have a Roboflow key, and no evidence established that this 1,092-image listing and the 1,097-row Hub mirror are identical.

For those reasons this project does **not** claim the Hub images are covered by Roboflow's CC BY 4.0 listing. To enable public model/Space release, obtain a rights-cleared dataset version (for example, an authorized Roboflow export with attribution requirements confirmed) or explicit permission from the image rights holders.

## Other pretrained weights

The transfer-learning model initializes from torchvision's MobileNetV3-Small ImageNet weights. Review torchvision's [model documentation](https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html) and upstream weight metadata before redistribution. The current repository does not publish trained weights.
