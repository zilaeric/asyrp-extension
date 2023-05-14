# Advancing Our Understanding of Diffusion Models: <br/> A Deep Dive Into "Diffusion Models Already Have a Semantic Latent Space"

### J. R. Gerbscheid, A. Ivășchescu, L. P. J. Sträter, E. Zila

---

In this blog post, we will discuss, reproduce and extend on the findings of the ICLR2023 paper titled "Diffusion Models Already Have a Semantic Latent Space". The paper introduces an algorithm called asymmetric reverse process (Asyrp) to uncover a semantic latent space in the bottleneck of frozen diffusion models. The authors showcase the effectiveness of Asyrp in the domain of image editing.

**Goal of this blog post:** The purpose of this blog post is threefold: 
1. Help other researchers understand the algorithm (Asyrp); 
2. Verify the authors' claims by reproducing the results; 
3. Extend on the discussion points of the paper.

## Image Editing Using Diffusion Models

Diffusion models (DM) can be effectively used for image editing, i.e., adding target attributes to real images. Multiple ways of achieving the task have been explored in the past, including image guidance \cite{choi2021ilvr, meng2021sdedit}, classifier guidance \cite{dhariwal2021diffusion, liu2023more}, and model fine-tuning \cite{kim2021diffusionclip}. These methods, however, fall short either because of the ambiguity and lack of control of the steering direction and magnitude of change, or because of the high computational costs they induce. Observing the capabilities of GANs that directly find editing directions from the latent space, hints that the discovery of such a space in diffusion models would provide great image editing capabilities. A way of obtaining it was suggested by Preechakul et al. \cite{preechakul2022diffusion}, who add to the reverse diffusion process the latent vector of the original image produced by an extra encoder. The problem of this model is that it can not use pretrained diffusion models, as they have to be trained with the added encoder. This is why Kwon et al. \cite{kwon2022diffusion} proposed Asyrp, which finds editing directions from the latent space in pretrained diffusion models.

Asyrp discovers semantic meaning in the bottleneck of the U-Net architecture. An augmentation to this bottleneck, $\mathbf{\Delta h_{t}}$, is predicted by a neural implicit function with inputs the bottleneck feature maps, $\mathbf{h_{t}}$, and the timestep $t$. This process of editing the bottleneck results in finding what the authors call the h-space, i.e. a semantic latent space that displays the following properties: homogeneity, linearity, robustness and consistency across timesteps.

Asyrp is trained to minimize the loss consisting of the directional CLIP loss and the reconstruction loss. To support the results of their method, Kwon et al. \cite{kwon2022diffusion} performed both qualitative and quantitative experiments. The metrics that they evaluated on are directional CLIP similarity and segmentation consistency.

In order to test the generalizability of the proposed Asyrp algorithm, we apply it to latent diffusion models (LDM) and also experiment with the network's architecture. Since LDMs currently represent the state-of-the-art in image generation \cite{rombach2021highresolution}, it is reasonable to investigate whether modifications in this h-space lead to meaningful attribute edits in the original images. Nonetheless, according to Park et al. \cite{park2023unsupervised} the semantic latent space of LDMs lacks structure and might be too complex for the methodology to be useful. On the other hand, we believe that by using a more complex architecture for the network that predicts the $\mathbf{\Delta h_{t}}$, we can capture intricate relationships in the data and achieve a significant performance boost over the original results. Another motivation for experimenting with this is the complexity of the latent space in LDMs, which suggests that attention-based networks might be better at finding relations between the dimensions.

## Recap on Diffusion Models

Over the past few years, we have observed a surge in popularity of generative models due to their proven ability to create realistic and novel content. DMs are a powerful new family of these models which has been shown to outperform other alternatives such as variational autoencoders (VAEs) and generative adversarial networks (GANs) on image synthesis \cite{dhariwal2021diffusion}. The basic idea behind them is to gradually add noise to the input data during the forward process and then train a neural network to recover the original data step-by-step in the reverse process. The Asyrp paper's authors chose to base their work on Denoising Diffusion Probabilistic Models (DDPM) \cite{nichol2021improved} and its successors, a widely-used algorithm that effectively implements this concept. In DDPMs the forward process $q$ is parameterized by a Markov process as shown in Equation 1, to produce latent variables $x_1$ through $x_T$ by adding Gaussian noise at each time step t with a variance of $\beta_t \in (0,1)$ following Equation 2.

$$\begin{align} 
q\left( x_1, \ldots, x_T \mid x_0 \right) := \prod_{t=1}^T q \left( x_t \mid x_{t-1} \right) & \qquad \qquad \text{(Equation 1)} \\ 
q\left( x_t \mid x_{t-1} \right) := \mathcal{N}\left( x_t ; \sqrt{1-\beta_t} x_{t-1}, \beta_t \mathbf{I} \right) & \qquad \qquad \text{(Equation 2)} 
\end{align}$$

To run the process in reverse starting from a sample $x_T \sim \mathcal{N}(0, \mathbf{I})$, the exact reverse distribution $q\left(x_{t-1} \mid x_t\right)$ needs to be approximated with Equation 3. This Markovian chain of slowly adding/removing noise is illustrated in Figure 1.

$$p_\theta \left( x_{t-1} \mid x_t \right) := \mathcal{N} \left( x_{t-1} ; \mu_\theta \left( x_t, t \right), \Sigma_\theta \left( x_t, t \right) \right) \qquad \qquad \text{(Equation 3)}$$

| ![Denoising process](figures/aprox.png) | 
|:-:| 
| **Figure 1.** The Markov process of diffusing noise and denoising \cite{ho2020denoising}. |

In DDPM $\mu_\theta\left(x_t, t\right)$ is estimated using a neural network that predicts the added noise $\epsilon$ at step $t$ as shown in Equation 4 and $\Sigma_\theta\left(x_t, t\right)$ is kept fixed to $\beta_t \mathbf{I}$. Then an efficient way to sample from an arbitrary step can be formulated as in Equation 5, with $v_T \sim \mathcal{N}(0, \mathbf{I})$ and $\alpha_t = \Pi_{s=1}^t \left( 1 - \beta_s \right)$.

$$\begin{align} 
\mu_\theta \left( x_t, t \right) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta\_t}{\sqrt{1 - \bar{\alpha}\_t}} \epsilon\_\theta \left( x_t, t \right) \right) & \qquad \qquad \text{(Equation 4)} \\ 
x_{t-1} = \frac{1}{\sqrt{1 - \beta_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \alpha_t}} \epsilon_\theta \left( x_t, t \right) \right) + \sqrt{\beta_t} v_t & \qquad \qquad \text{(Equation 5)}
\end{align}$$

One major improvement on this algorithm was the Denoising Diffusion Implicit Model (DDIM) \cite{song2020denoising}. In DDIM an alternative non-Markovian noising process is used instead of Equation 1 as shown in Equation 6. Down the line this leads to a change in the way an arbitrary step is sampled in the reverse process to Equation 7, with $\sigma_t=\eta \sqrt{\left(1-\alpha_{t-1}\right) /\left(1-\alpha_t\right)} \sqrt{1-\alpha_t / \alpha_{t-1}}$ and $\eta$ a hyper-parameter.

$$\begin{align} 
q_\sigma\left(x_{t-1} \mid x_t, x_0\right)=\mathcal{N}\left(\sqrt{\alpha_{t-1}} x_0+\sqrt{1-\alpha_{t-1}-\sigma_t^2} \cdot \frac{x_t-\sqrt{\alpha_t} x_0}{\sqrt{1-\alpha_t}}, \sigma_t^2 \boldsymbol{I} \right) & \qquad \qquad \text{(Equation 6)} \\ 
x_{t-1} = \sqrt{\alpha_{t-1}} \left( \frac{x_t - \sqrt{1 - \alpha_t} \epsilon_\theta \left( x_t, t \right)}{\sqrt{\alpha_t}} \right) + \sqrt{1-\alpha\_{t-1}-\sigma_t^2} \cdot \epsilon\_\theta \left( x_t, t \right) + \sigma_t v_t & \qquad \qquad \text{(Equation 7)}
\end{align}$$

Equation 7 was the starting point for the Asyrp paper, however they reformulated it as shown in Equation 8. Why this is convenient will become apparent in the next section. In this formulation $\textbf{P}_t$ can be viewed as the predicted $x_0$ and $\textbf{D}_t$ as the direction pointing to $x_t$.

$$x_{t-1} = \sqrt{\alpha_{t-1}} \mathbf{P}\_t \left( \epsilon\_\theta \left( x_t, t \right) \right) + \mathbf{D}\_t \left( \epsilon\_\theta \left( x_t, t \right) \right) + \sigma_t v_t \qquad \qquad \text{(Equation 8)}$$

In practise this boils down to training one neural network $\epsilon\_\theta \left( x_t, t \right)$ \cite{ho2020denoising}, with (image $x_0$, time-step $t$) pairs. Then because the noising schedule is known we can add noise in one go to $x_0$ to get $x_t$ and $x_{t+1}$. Finally with Equation 9 the loss can be calculated between the actually added $\epsilon$ between $x_t$ and $x_{t+1}$ and the predicted $\epsilon$. 

$$L_{D M} = \mathbb{E}\_{ x, \epsilon \sim \mathcal{N}(0, 1), t } \left\[ \left\| \epsilon - \epsilon\_\theta \left( x_t, t \right) \right\|_2^2 \right\] \qquad \qquad \text{(Equation 9)}$$

Lastly in this blog post we will extend Asyrp to show that the same procedure can be applied to latent diffusion models \cite{rombach2021highresolution}. Conveniently DDIM is also the algorithm behind latent diffusion models, however the diffusion process runs in the latent space instead of the pixel space. A sperate autoencoder is trained, where the encoder $\mathcal{E}$ is used to compress the image $x_0$ to a smaller latent vector $z_0$ and the decoder $\mathcal{D}$ is used to reconstruct the image $\hat{x}_0$ from the computed latent vector $\hat{z}_0$. For the rest all the steps are followed as above, but replacing $x$ by $z$. This leads to training a neural network $\epsilon\_\theta \left( z_t, t \right)$ and optimizing it with the following loss: 

$$L_{L D M} := \mathbb{E}\_{ \mathcal{E}(x), \epsilon \sim \mathcal{N}(0, 1), t } \left\[ \left\| \epsilon - \epsilon\_\theta \left( z_t, t \right) \right\|_2^2 \right\] \qquad \qquad \text{(Equation 10)}$$

## Discovering Semantic Latent Space

This returns us to the original goal of the Asyrp paper, i.e. to manipulate the semantic latent space of images generated from Gaussian noise with a \textbf{pretrained and frozen diffusion model} to edit them. To achieve this the authors propose an asymmetric reverse process (Asyrp) in which they alter the way an arbitrary step is sampled in the reverse process to Equation 11.

$$x_{t-1} = \sqrt{\alpha\_{t-1}} \mathbf{P}\_t \left( \tilde{\epsilon}\_\theta \left( x_t, t \right) \right) + \mathbf{D}\_t \left( \epsilon\_\theta \left( x_t, t \right) \right) + \sigma_t v_t \qquad \qquad \text{(Equation 11)}$$

As can be seen the noise estimate used to predict $x_0$ is edited while the direction towards $x_t$ stays unchanged so that $x_{t-1}$ follows the original flow at each time-step. The idea is that by doing this low level information will change, while high level details stay the same. For example, the location of the eyebrows is different, but they are the same eyebrows.

But that raises an important question: How to edit the predicted noise in a meaningful way such that the change in the image reflects the semantic change that the user wants? 

In practise, all SOTA diffusion models use the U-net architecture to approximate $\epsilon_\theta\left(x_t, t\right)$. The authors therefor propose an augmentation to the bottleneck of the U-net, $\Delta h_{t}$, which is predicted by a neural network with inputs the bottleneck feature maps, $h_{t}$, the timestep $t$, and importantly also a representation of the semantic change that the user desires. More formally, this leads to sampling a step in the reverse process following Equation 12, where $\epsilon_{\theta}(x_t, t |\Delta h_t)$ adds $\Delta h_t$ to the original feature map $h_t$.

$$x_{t-1} = \sqrt{\alpha\_{t-1}} \mathbf{P}\_t \left( \epsilon\_\theta \left( x_t, t \mid \Delta h_t \right) \right) + \mathbf{D}\_t \left( \epsilon\_\theta \left( x_t, t \right) \right) + \sigma_t v_t \qquad \qquad \text{(Equation 12)}$$

The neural network, $f_t$, used for predicting $\Delta h_{t}$ is trained to edit $h_t$ in such a way that the semantics of $x_{t-1}$ change according to the users prompt. In the Asyrp paper, a pretrained CLIP model is used for the text-driven image editing. 

CLIP (Contrastive Language-Image Pretraining) \cite{radford2021learning} is a multi-modal, zero-shot model that predicts the most relevant caption for an image. It consists of a text encoder and an image encoder (both relying on a transformer architecture) that encode the data into a multimodal embedding space. The encoders are jointly trained on a dataset of images and their true textual descriptions, using a contrastive loss function. This loss function aims to maximize the cosine similarity of images and their corresponding text and minimize the similarity between images and texts that do not occur together. 

For the neural network used for predicting $\Delta h_{t}$ this boils down to training to minimize the directional CLIP loss shown in Equation 13 and the difference between the predicted and the original image. Both the reference and the generated images are embedded into CLIP-space and the directional loss requires the vector that connects them, $\Delta I = E_I(\mathbf{x}\_{edit}) - E_I(\mathbf{x}\_{ref})$, to be parallel to the one that connects the reference and the target text, $\Delta T = E_T(y_{target}) - E_T(y_{ref})$:

$$\mathcal{L}\_{direction} (\mathbf{x}\_{edit}, y\_{target}; \mathbf{x}\_{ref}, y\_{ref}) = 1 - \frac{\Delta I \cdot \Delta T}{\Vert \Delta I \Vert \Vert \Delta T \Vert} \qquad \qquad \text{(Equation 13)}$$

This leads to the loss function that Asyrp is trained to minimize in Equation 14, where $\mathbf{P}^{\text{edit}}\_t$ replaces $\mathbf{x}\_{edit}$ which is the predicted $\mathbf{x}\_{0}$ at timestep $t$, $\mathbf{P}^{\text{ref}}\_t$ replaces $\mathbf{x}\_{ref}$ which is the original, and $\lambda\_{\text{CLIP}}$ and $\lambda\_{\text{recon}}$ are weight parameters for each loss:

$$\mathcal{L}\_t = \lambda\_{\text{CLIP}} \mathcal{L}\_{direction} (\mathbf{P}^{\text{edit}}\_t, y^{target}; \mathbf{P}^{\text{ref}}\_t, y^{ref}) + \lambda\_{\text{recon}} | \mathbf{P}^{\text{edit}}\_t - \mathbf{P}^{\text{ref}}\_t | \qquad \qquad \text{(Equation 14)}$$

Figure 2 visualizes the generative process of Asyrp intuitively. As shown by the green box on the left, the process only changes $\textbf{P}\_t$ while preserving $\textbf{D}\_t$. On the right side, the figure illustrates how Asyrp alters the reverse process to achieve the desired outcome by adjusting the attributes in the h-space. However, in practise they also make use of some practical tricks to make the theory work. Foremost, they only edit the h-space in an empirically found window which is for most examples around the first 30\% time-steps of the reverse process. Secondly, they scale $\Delta h_{t}$ using non-accelerated sampling. Lastly, they make use of a technique called quality boosting in roughly the last 30\% time-steps. All these techniques are explained more thoroughly in the paper, but not essential for the intends and purposes of this blog post.

| ![Asyrp](figures/asyrp.png) | 
|:-:| 
| **Figure 2.** Asymmetric reverse process. |

Practically, $f_t$ is implemented as shown in Figure 3. However, the authors note that they haven't explored with other network architectures. That let us to experiment further, which eventually led the network architecture in Figure 4. **TO-DO**

| ![Asyrp architecture](figures/architecture_asyrp.png) | 
|:-:| 
| **Figure 3.** Architecture of $f_t$ in the Asyrp paper \cite{kwon2022diffusion}. |

| ![Asyrp proposed architecture](figures/architecture_asyrp.png) | 
|:-:| 
| **Figure 4.** **TO-DO:** Architecture of our $f_t$. |

Lastly, to apply this methodology to LDMs two things should be noted. Firstly, Equation 12 can be easily altered to Equation 15 to sample steps in the reverse process. However, the same can not be said for the directional CLIP loss in Equation 13. Here the both the reference and the generated image are needed to calculate the loss, but the whole point of LDMs is that you do not calculate those every step. Nevertheless, to use the Asyrp algorithm they can be computed by running the decoder $\mathcal{D}$ on $z_t$ at every time-step.

$$z_{t-1} = \sqrt{\alpha\_{t-1}} \mathbf{P}\_t \left( \epsilon\_\theta \left( z_t, t \mid \Delta h_t \right) \right) + \mathbf{D}\_t \left( \epsilon\_\theta \left( z_t, t \right) \right) + \sigma_t v_t \qquad \qquad \text{(Equation 15)}$$

## Evaluating Diffusion Models

In order to evaluate the performance of diffusion models when it comes to image editing, besides qualitative results and conducting user studies \cite{kwon2022diffusion}\cite{kim2021diffusionclip}, the following metrics are generally used: Directional CLIP similarity ($S_{dir}$), segmentation-consistency (SC), Fr\'echet Inception Distance (FID) and face identity similarity (ID). The Asyrp paper uses $S_{dir}$ and SC to compare its performance to DiffusionCLIP, which in turn shows that it outperforms both StyleCLIP \cite{patashnik2021styleclip} and StyleGAN-NADA \cite{gal2022stylegan} in $S_{dir}$, SC and ID.

The directional CLIP similarity score measures how well does the diffusion model preserve the direction of gradients in an image after editing. It is mathematically computed as $1 - \mathcal{L}\_{direction}$, where $\mathcal{L}\_{direction}$ is the directional CLIP loss from Equation \ref{clip-dir-loss}. The higher the score, the better image editing performance of the model.

Semantic consistency is a metric that has been introduced in order to evaluate the consistency of network predictions on video sequences. In the image editing setting, it compares the segmentation maps of the reference and the edited image by computing the mean intersection over union of the two. Knowing this, we can reason that high SC scores to not necessarily mean good image content modification, as it can be seen in Figure \ref{fig:sc}. This is an example that clearly shows how this metric fails on evaluating editing performance. The DiffusionCLIP model tries to preserve structure and shape in the image, while Asyrp allows more changes that lead to desired attribute alterations.

| ![Segmentation consistency](figures/sc.png) | 
|:-:| 
| **Figure 5.** Segmentation masks of the Reference image, Asyrp and DiffustionCLIP generated images for computing SC for the attribute smiling \cite{kwon2022diffusion}. |

The ID score measures how well the identity of a face has been preserved after editing. It uses the pre-trained ArcFace face recognition model \cite{deng2019arcface} in order to generate the feature vectors of the original and the edited faces, and then computes the cosine similarity between them. 

The FID metric compares the distribution of the edited images with the distribution of the reference ones in feature space. Lower FID scores correspond to better image editing. In order to compute the FID score, the activations of the last layer prior to the output classification one of the Inception v3 model are determined for a set of edited and source images. The mean and the covariance of the activations is computed, so they can be modelled as multivariate Gaussians: $\mathcal{N}(\mu, \Sigma)$ being the distribution of the edited images' features and $\mathcal{N}(\mu_{ref}, \Sigma_{ref})$ the distribuiton of the reference images. The FID is then calculated using Equation \ref{fid}:

$$FID = \Vert \mu - \mu\_{ref} \Vert_2^2 + tr \left( \Sigma + \Sigma\_{ref} - 2 { \left( \Sigma^\frac{1}{2} \Sigma\_{ref} \Sigma^\frac{1}{2} \right) }^\frac{1}{2} \right) \qquad \qquad \text{(Equation 16)}$$

## Reproduction of the Experiments

We begin by reproducing the qualitative and quantitative results of the original paper. We conduct our reproducibility experiments on the CelebA-HQ \cite{karras2017progressive} dataset and use the ...... diffusion model \cite{}. We make use of the [open source code](https://github.com/kwonminki/Asyrp_official/tree/main/models) from the original paper to do so.

Figures 6 and 7 show that the results obtained in the original Asyrp paper are reproducible and that editing in the h-space results in high performance image generation for both in and unseen (attributes that are not included in the training dataset) domains. 

| ![In-domain](figures/in.png) | 
|:-:| 
| **Figure 6.** **TO-DO:** Editing results of Asyrp on CelebA-HQ for in-domain attributes. |

| ![Unseen-domain](figures/unseen.png) | 
|:-:| 
| **Figure 7.** **TO-DO:** Editing results of Asyrp on CelebA-HQ for unseen-domain attributes. |

To quantitatively appreciate the performance of the Asyrp model, we reproduce the evaluation they conducted and compute the Directional CLIP score for the same three in-domain attributes (smiling, sad, tanned) and two unseen-domain attributes (Pixar, Neanderthal) on a set of 100 images per attribute from the CelebA-HQ dataset. The available repository does not provide code for implementing neither of the evaluation metrics, which leads to also not knowing which 100 images from the dataset were considered when computing the scores. We took the first 100 images and the comparative results can be seen in Table . We did not implement the segmentation consistency score, as we showed in the Evaluation Diffusion Models section that it has shortcomings, but we computed the FID score that is more meaningful in the case of image editing. % here add a table with the results; also maybe add ID score because there is already an implementation

We also conducted reproducibility experiments on the linearity and consistency across timesteps of the model. The results can be seen in Figures 8 and 9.

| ![Linearity](figures/linearity.png) | 
|:-:| 
| **Figure 8.** **TO-DO:** Linearity of h-space. |

| ![Linear combinations](figures/combination.png) | 
|:-:| 
| **Figure 9.** **TO-DO:** Linear combination of attributes. |

## Further Analysis

## Conclusion and Future Research Directions

## Bibliography
