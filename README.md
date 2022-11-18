# Alexa Teacher Models

This repo includes artifacts related to the Alexa Teacher Model initiative. Please check back for updates!

## AlexaTM 20B

AlexaTM 20B is a 20B-Parameter sequence-to-sequence transformer model created by the Alexa Teacher Model (AlexaTM) team at Amazon. The model was trained on a mixture of Common Crawl (mC4) and Wikipedia data across 12 languages using denoising and Causal Language Modeling (CLM) tasks.

AlexaTM 20B can be used for in-context learning. "In-context learning," also known as "prompting," refers to a method for using NLP models in which no fine tuning is required per task. Training examples are provided to the model only as part of the prompt given as inference input, a paradigm known as "few-shot in-context learning." In some cases, the model can perform well without any training data at all, a paradigm known as "zero-shot in-context learning."

To learn more about the model, please read the [Amazon Science blog post](https://www.amazon.science/blog/20b-parameter-alexa-model-sets-new-marks-in-few-shot-learning) and the [paper](https://arxiv.org/abs/2208.01448).

The model is currently available for noncommercial use via SageMaker JumpStart. The model can be accessed using the following steps:

1. [Create](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/) an AWS account if needed.
1. In your AWS account, search for `SageMaker` in the search bar and click on it.
1. Once in the SageMaker experience, create a [domain](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-studio-onboard.html) and a studio user if none yet exist. All of the default settings can be used.
1. In the control panel, click `Launch app` next to the user you wish to use. Launch a studio instance.
1. Once in the studio, there will be a launcher showing JumpStart as one of the tiles. Click `Go to SageMaker Jumpstart`. Alternatively, JumpStart can be accessed by 3-pointed orange symbol on the far left of the studio.
1. Once in JumpStart, click the `Notebooks` button.
1. Browse or search for our example notebook entitled `In-context learning with AlexaTM 20B`.
1. There will be a button at the top to copy the read-only version into your studio.
1. Ensure that your kernel has started, and run the notebook.

Note: You can also find our example notebook [here](https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/jumpstart_alexatm20b/Amazon_Jumpstart_AlexaTM_20B.ipynb)

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project (the code) is licensed under the Apache-2.0 License. The AlexaTM 20B model weights are licensed under the [Alexa Teacher Model License Agreement](MODEL_LICENSE.md).
