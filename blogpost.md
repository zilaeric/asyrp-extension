# Advancing Our Understanding of Diffusion Models: <br/> A Deep Dive Into "Diffusion Models Already Have a Semantic Latent Space"

### J. R. Gerbscheid, A. Ivășchescu, L. P. J. Sträter, E. Zila

---

In this blog post, we will discuss, reproduce and extend on the findings of the ICLR2023 paper titled ["Diffusion Models Already Have a Semantic Latent Space"](https://arxiv.org/abs/2210.10960). The paper introduces an algorithm called asymmetric reverse process (Asyrp) to uncover a semantic latent space in the bottleneck of frozen diffusion models. The authors showcase the effectiveness of Asyrp in the domain of image editing.

**Goal of this blog post:** The purpose of this blog post is threefold: 
1. Help other researchers understand the algorithm (Asyrp). 
2. Verify the authors' claims by reproducing the results. 
3. Extend on the discussion points of the paper.

## Image Editing Using Diffusion Models

Diffusion models (DM) can be effectively used for image editing, i.e., adding target attributes to real images. Multiple ways of achieving the task have been explored in the past, including image guidance \[1, 10\], classifier guidance \[3, 9\], and model fine-tuning \[7\]. These methods, however, fall short either because of the ambiguity and lack of control of the steering direction and magnitude of change, or because of the high computational costs they induce. Observing the capabilities of GANs that directly find editing directions from the latent space, hints that the discovery of such a space in diffusion models would provide great image editing capabilities. A way of obtaining it was suggested by Preechakul et al. \[14\], who add to the reverse diffusion process the latent vector of the original image produced by an extra encoder. The problem of this model is that it can not use pretrained diffusion models, as they have to be trained with the added encoder. This is why Kwon et al. \[8\] proposed Asyrp, which finds editing directions from the latent space in pretrained diffusion models.

Asyrp discovers semantic meaning in the bottleneck of the U-Net architecture. An augmentation to this bottleneck, $\mathbf{\Delta h_{t}}$, is predicted by a neural implicit function with inputs the bottleneck feature maps, $\mathbf{h_{t}}$, and the timestep $t$. This process of editing the bottleneck results in finding what the authors call the h-space, i.e. a semantic latent space that displays the following properties: homogeneity, linearity, robustness and consistency across timesteps.

Asyrp is trained to minimize the loss consisting of the directional CLIP loss and the reconstruction loss. To support the results of their method, Kwon et al. \[8\] performed both qualitative and quantitative experiments. The metrics that they evaluated on are directional CLIP similarity and segmentation consistency.

In order to test the performance and the generalizability of the proposed Asyrp algorithm, we reproduce their main 
qualitative and quantitative experiments on the 
CelebA-HQ \[6\] dataset, introduce a new metric, the FID score, change the architecture of the neural network that 
produces the semantic latent space used for editing into a transformer-based network and perform an ablation study on it.
Also, since LDMs currently represent the state-of-the-art in image generation \[16\], it is reasonable to investigate 
whether this method could be applied to them and whether it would lead to meaningful attribute edits in the original images.
## <a name="recap">Recap on Diffusion Models</a>

Over the past few years, we have observed a surge in popularity of generative models due to their proven ability to create realistic and novel content. DMs are a powerful new family of these models which has been shown to outperform other alternatives such as variational autoencoders (VAEs) and generative adversarial networks (GANs) on image synthesis \[3\]. The basic idea behind them is to gradually add noise to the input data during the forward process and then train a neural network to recover the original data step-by-step in the reverse process. The Asyrp paper's authors chose to base their work on Denoising Diffusion Probabilistic Models (DDPM) \[11\] and its successors, a widely-used algorithm that effectively implements this concept. In DDPMs the forward process $q$ is parameterized by a Markov process as shown in Equation 1, to produce latent variables $x_1$ through $x_T$ by adding Gaussian noise at each time step t with a variance of $\beta_t \in (0,1)$ following Equation 2.

$$\begin{align} 
q\left( x_1, \ldots, x_T \mid x_0 \right) := \prod_{t=1}^T q \left( x_t \mid x_{t-1} \right) & \qquad \qquad \text{(Equation 1)} \\ 
q\left( x_t \mid x_{t-1} \right) := \mathcal{N}\left( x_t ; \sqrt{1-\beta_t} x_{t-1}, \beta_t \mathbf{I} \right) & \qquad \qquad \text{(Equation 2)} 
\end{align}$$

To run the process in reverse starting from a sample $x_T \sim \mathcal{N}(0, \mathbf{I})$, the exact reverse distribution $q\left(x_{t-1} \mid x_t\right)$ needs to be approximated with Equation 3. This Markovian chain of slowly adding/removing noise is illustrated in Figure 1.

$$p_\theta \left( x_{t-1} \mid x_t \right) := \mathcal{N} \left( x_{t-1} ; \mu_\theta \left( x_t, t \right), \Sigma_\theta \left( x_t, t \right) \right) \qquad \qquad \text{(Equation 3)}$$

| ![Denoising process](figures/aprox.png) | 
|:-:| 
| **Figure 1.** The Markov process of diffusing noise and denoising \[5\]. |

In DDPM $\mu_\theta\left(x_t, t\right)$ is estimated using a neural network that predicts the added noise $\epsilon$ at step $t$ as shown in Equation 4 and $\Sigma_\theta\left(x_t, t\right)$ is kept fixed to $\beta_t \mathbf{I}$. Then an efficient way to sample from an arbitrary step can be formulated as in Equation 5, with $v_T \sim \mathcal{N}(0, \mathbf{I})$ and $\alpha_t = \Pi_{s=1}^t \left( 1 - \beta_s \right)$.

$$\begin{align} 
\mu_\theta \left( x_t, t \right) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta\_t}{\sqrt{1 - \bar{\alpha}\_t}} \epsilon\_\theta \left( x_t, t \right) \right) & \qquad \qquad \text{(Equation 4)} \\ 
x_{t-1} = \frac{1}{\sqrt{1 - \beta_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \alpha_t}} \epsilon_\theta \left( x_t, t \right) \right) + \sqrt{\beta_t} v_t & \qquad \qquad \text{(Equation 5)}
\end{align}$$

One major improvement on this algorithm was the Denoising Diffusion Implicit Model (DDIM) \[17\]. In DDIM an alternative non-Markovian noising process is used instead of Equation 1 as shown in Equation 6. Down the line this leads to a change in the way an arbitrary step is sampled in the reverse process to Equation 7, with $\sigma_t=\eta \sqrt{\left(1-\alpha_{t-1}\right) /\left(1-\alpha_t\right)} \sqrt{1-\alpha_t / \alpha_{t-1}}$ and $\eta$ a hyper-parameter.

$$\begin{align} 
q_\sigma\left(x_{t-1} \mid x_t, x_0\right)=\mathcal{N}\left(\sqrt{\alpha_{t-1}} x_0+\sqrt{1-\alpha_{t-1}-\sigma_t^2} \cdot \frac{x_t-\sqrt{\alpha_t} x_0}{\sqrt{1-\alpha_t}}, \sigma_t^2 \boldsymbol{I} \right) & \qquad \qquad \text{(Equation 6)} \\ 
x_{t-1} = \sqrt{\alpha_{t-1}} \left( \frac{x_t - \sqrt{1 - \alpha_t} \epsilon_\theta \left( x_t, t \right)}{\sqrt{\alpha_t}} \right) + \sqrt{1-\alpha\_{t-1}-\sigma_t^2} \cdot \epsilon\_\theta \left( x_t, t \right) + \sigma_t v_t & \qquad \qquad \text{(Equation 7)}
\end{align}$$

Equation 7 was the starting point for the Asyrp paper, however they reformulated it as shown in Equation 8. Why this is convenient will become apparent in the next section. In this formulation $\textbf{P}_t$ can be viewed as the predicted $x_0$ and $\textbf{D}_t$ as the direction pointing to $x_t$.

$$x_{t-1} = \sqrt{\alpha_{t-1}} \mathbf{P}\_t \left( \epsilon\_\theta \left( x_t, t \right) \right) + \mathbf{D}\_t \left( \epsilon\_\theta \left( x_t, t \right) \right) + \sigma_t v_t \qquad \qquad \text{(Equation 8)}$$

In practice, this boils down to training one neural network $\epsilon\_\theta \left( x_t, t \right)$ \[5\], with (image $x_0$, time-step $t$) pairs. Then because the noising schedule is known we can add noise in one go to $x_0$ to get $x_t$ and $x_{t+1}$. Finally with Equation 9 the loss can be calculated between the actually added $\epsilon$ between $x_t$ and $x_{t+1}$ and the predicted $\epsilon$. 

$$L_{D M} = \mathbb{E}\_{ x, \epsilon \sim \mathcal{N}(0, 1), t } \left\[ \left\| \epsilon - \epsilon\_\theta \left( x_t, t \right) \right\|_2^2 \right\] \qquad \qquad \text{(Equation 9)}$$

> **Note**
> For a thorough introduction to Diffusion Models we would like to highlight an outstanding [blog post by Lilian Weng](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/).

## <a name="discover">Discovering Semantic Latent Space</a>

This returns us to the original goal of the Asyrp paper, i.e. to manipulate the semantic latent space of images generated from Gaussian noise with a **pretrained and frozen diffusion model** to edit them. To achieve this the authors propose an asymmetric reverse process (Asyrp) in which they alter the way an arbitrary step is sampled in the reverse process to Equation 10.

$$x_{t-1} = \sqrt{\alpha\_{t-1}} \mathbf{P}\_t \left( \tilde{\epsilon}\_\theta \left( x_t, t \right) \right) + \mathbf{D}\_t \left( \epsilon\_\theta \left( x_t, t \right) \right) + \sigma_t v_t \qquad \qquad \text{(Equation 10)}$$

As can be seen the noise estimate used to predict $x_0$ is edited while the direction towards $x_t$ stays unchanged so that $x_{t-1}$ follows the original flow at each time-step. The idea is that by doing this low level information will change, while high level details stay the same. For example, the location of the eyebrows is different, but they are the same eyebrows.

But that raises an important question: How to edit the predicted noise in a meaningful way such that the change in the image reflects the semantic change that the user wants? 

In practise, all SOTA diffusion models use the U-net architecture to approximate $\epsilon_\theta\left(x_t, t\right)$. The authors therefor propose an augmentation to the bottleneck of the U-net, $\Delta h_{t}$, which is predicted by a neural network with inputs the bottleneck feature maps, $h_{t}$, the timestep $t$, and importantly also a representation of the semantic change that the user desires. More formally, this leads to sampling a step in the reverse process following Equation 11, where $\epsilon_{\theta}(x_t, t |\Delta h_t)$ adds $\Delta h_t$ to the original feature map $h_t$.

$$x_{t-1} = \sqrt{\alpha\_{t-1}} \mathbf{P}\_t \left( \epsilon\_\theta \left( x_t, t \mid \Delta h_t \right) \right) + \mathbf{D}\_t \left( \epsilon\_\theta \left( x_t, t \right) \right) + \sigma_t v_t \qquad \qquad \text{(Equation 11)}$$

The neural network, $f_t$, used for predicting $\Delta h_{t}$ is trained to edit $h_t$ in such a way that the semantics of $x_{t-1}$ change according to the users prompt. In the Asyrp paper, a pretrained CLIP model is used for the text-driven image editing. 

CLIP (Contrastive Language-Image Pretraining) \[15\] is a multi-modal, zero-shot model that predicts the most relevant caption for an image. It consists of a text encoder and an image encoder (both relying on a transformer architecture) that encode the data into a multimodal embedding space. The encoders are jointly trained on a dataset of images and their true textual descriptions, using a contrastive loss function. This loss function aims to maximize the cosine similarity of images and their corresponding text and minimize the similarity between images and texts that do not occur together. 

For the neural network used for predicting $\Delta h_{t}$ this boils down to training to minimize the directional CLIP loss shown in Equation 12 and the difference between the predicted and the original image. Both the reference and the generated images are embedded into CLIP-space and the directional loss requires the vector that connects them, $\Delta I = E_I(\mathbf{x}\_{edit}) - E_I(\mathbf{x}\_{ref})$, to be parallel to the one that connects the reference and the target text, $\Delta T = E_T(y_{target}) - E_T(y_{ref})$:

$$\mathcal{L}\_{direction} (\mathbf{x}\_{edit}, y\_{target}; \mathbf{x}\_{ref}, y\_{ref}) = 1 - \frac{\Delta I \cdot \Delta T}{\Vert \Delta I \Vert \Vert \Delta T \Vert} \qquad \qquad \text{(Equation 12)}$$

This leads to the loss function that Asyrp is trained to minimize in Equation 13, where $\mathbf{P}^{\text{edit}}\_t$ replaces $\mathbf{x}\_{edit}$ which is the predicted $\mathbf{x}\_{0}$ at timestep $t$, $\mathbf{P}^{\text{ref}}\_t$ replaces $\mathbf{x}\_{ref}$ which is the original, and $\lambda\_{\text{CLIP}}$ and $\lambda\_{\text{recon}}$ are weight parameters for each loss:

$$\mathcal{L}\_t = \lambda\_{\text{CLIP}} \mathcal{L}\_{direction} (\mathbf{P}^{\text{edit}}\_t, y^{target}; \mathbf{P}^{\text{ref}}\_t, y^{ref}) + \lambda\_{\text{recon}} | \mathbf{P}^{\text{edit}}\_t - \mathbf{P}^{\text{ref}}\_t | \qquad \qquad \text{(Equation 13)}$$

Figure 2 visualizes the generative process of Asyrp intuitively. As shown by the green box on the left, the process only changes $\textbf{P}\_t$ while preserving $\textbf{D}\_t$. On the right side, the figure illustrates how Asyrp alters the reverse process to achieve the desired outcome by adjusting the attributes in the h-space. However, in practise they also make use of some practical tricks to make the theory work. Foremost, they only edit the h-space in an empirically found window which is for most examples around the first 30\% time-steps of the reverse process. Secondly, they scale $\Delta h_{t}$ using non-accelerated sampling. Lastly, they make use of a technique called quality boosting in roughly the last 30\% time-steps. All these techniques are explained more thoroughly in the paper, but not essential for the intends and purposes of this blog post.

| ![Asyrp](figures/asyrp.png) | 
|:-:| 
| **Figure 2.** Asymmetric reverse process. |

## <a name="architecture">Model Architecture</a>
**Section Ana & Jonathan with figures and short description why for all the tried archtectures, adding, multiplying, Ada something etc**

Practically, $f_t$ is implemented as shown in Figure 3. However, the authors note that they haven't explored with other network architectures. That let us to experiment further, which eventually led the network architecture in Figure 4. **TO-DO**

| ![Asyrp architecture](figures/architecture_asyrp.png) | 
|:-:| 
| **Figure 3.** Architecture of $f_t$ in the Asyrp paper \[8\]. |

| ![Asyrp proposed architecture](figures/architecture_asyrp.png) | 
|:-:| 
| **Figure 4.** **TO-DO:** Architecture of our $f_t$. |

## Evaluating Diffusion Models

In order to evaluate the performance of diffusion models when it comes to image editing, besides qualitative results and conducting user studies \[8, 7\], the following metrics are generally used: Directional CLIP similarity ($S_{dir}$), segmentation-consistency (SC), Fr\'echet Inception Distance (FID), and face identity similarity (ID). The Asyrp paper uses $S_{dir}$ and SC to compare its performance to DiffusionCLIP, which in turn shows that it outperforms both StyleCLIP \[13\] and StyleGAN-NADA \[4\] in $S_{dir}$, SC, and ID.

The directional CLIP similarity score measures how well the diffusion model preserves the direction of gradients in an image after editing. It is mathematically computed as $1 - \mathcal{L}\_{direction}$, where $\mathcal{L}\_{direction}$ is the directional CLIP loss from Equation 12. The higher the score, the better image editing performance of the model.

Semantic consistency is a metric that has been introduced in order to evaluate the consistency of network predictions on video sequences. In the image editing setting, it compares the segmentation maps of the reference and the edited image by computing the mean intersection over the union of the two. Knowing this, we can reason that high SC scores do not necessarily mean good image content modification, as can be seen in Figure 5. This is an example that clearly shows how this metric fails on evaluating editing performance. The DiffusionCLIP model tries to preserve structure and shape in the image, while Asyrp allows more changes that lead to desired attribute alterations.

| ![Segmentation consistency](figures/sc.png) | 
|:-:| 
| **Figure 5.** Segmentation masks of the Reference image, Asyrp and DiffustionCLIP generated images for computing SC for the attribute smiling \[8\]. |

The ID score measures how well the identity of a face has been preserved after editing. It uses the pre-trained ArcFace face recognition model \[2\] in order to generate the feature vectors of the original and the edited faces, and then computes the cosine similarity between them. 

The FID metric compares the distribution of the edited images with the distribution of the referential images in a feature space. Lower FID scores correspond to better image editing. In order to compute the image features, one commonly employs the Inception-v3 model \[18\]. In particular, the model's activations of the last layer prior to the output classification layer are calculated for a set of edited and source images. The mean and the covariance of the activations is computed, so they can be modelled as multivariate Gaussians: $\mathcal{N}(\mu, \Sigma)$ being the distribution of the edited images' features and $\mathcal{N}(\mu_{ref}, \Sigma_{ref})$ the distribution of the reference images' features. The FID is then calculated as follows:

$$FID = \Vert \mu - \mu\_{ref} \Vert_2^2 + tr \left( \Sigma + \Sigma\_{ref} - 2 { \left( \Sigma^\frac{1}{2} \Sigma\_{ref} \Sigma^\frac{1}{2} \right) }^\frac{1}{2} \right). \qquad \qquad \text{(Equation 14)}$$

## Reproduction of the Experiments

We begin by reproducing the qualitative and quantitative results of the original paper. To sustain the limits of our computational budget, we restrict our reproduction efforts to the CelebA-HQ \[6\] dataset. Our experiments are based on the [original implementation](https://github.com/kwonminki/Asyrp_official/tree/main/models), however, we found that some of the features required for successful reproduction, especially those relating to quantitative evaluation, are missing from the repository. Generally, we follow the computational set-up specified by the original authors in full. Specifically, we use hyperparameter values as specified in Table 1, which were recovered from \[8, Table 2\] and \[8, Table 3\]. For most of our experiments, we use 40 time steps during both the inversion and generation phase of training and inference. We model checkpoints trained for a single iteration over all images in the training sample.

<table align="center">
  <tr align="center">
      <th align="left">$y_{ref}$</th>
      <th align="left">$y_{target}$</th>
      <th>$\lambda_{\text{CLIP}}$</th>
      <th>$\lambda_{\text{recon}}$</th>
      <th>$t_{\text{edit}}$</th>
      <th>$t_{\text{boost}}$</th>
      <th>domain</th>
  </tr>
  <tr align="center">
    <td align="left">face</td>
    <td align="left">smiling face</td>
    <td>0.8</td>
    <td>3*0.899</td>
    <td>513</td>
    <td>167</td>
    <td>IN</td>
  </tr>
  <tr align="center">
    <td align="left">face</td>
    <td align="left">sad face</td>
    <td>0.8</td>
    <td>3*0.894</td>
    <td>513</td>
    <td>167</td>
    <td>IN</td>
  </tr>
  <tr align="center">
    <td align="left">face</td>
    <td align="left">angry face</td>
    <td>0.8</td>
    <td>3*0.892</td>
    <td>512</td>
    <td>167</td>
    <td>IN</td>
  </tr>
  <tr align="center">
    <td align="left">face</td>
    <td align="left">tanned face</td>
    <td>0.8</td>
    <td>3*0.886</td>
    <td>512</td>
    <td>167</td>
    <td>IN</td>
  </tr>
  <tr align="center">
    <td align="left">a person</td>
    <td align="left">a man</td>
    <td>0.8</td>
    <td>3*0.910</td>
    <td>513</td>
    <td>167</td>
    <td>IN</td>
  </tr>
  <tr align="center">
    <td align="left">a person</td>
    <td align="left">a woman</td>
    <td>0.8</td>
    <td>3*0.891</td>
    <td>513</td>
    <td>167</td>
    <td>IN</td>
  </tr>
  <tr align="center">
    <td align="left">person</td>
    <td align="left">young person</td>
    <td>0.8</td>
    <td>3*0.905</td>
    <td>515</td>
    <td>167</td>
    <td>IN</td>
  </tr>
  <tr align="center">
    <td align="left">person</td>
    <td align="left">person with curly hair</td>
    <td>0.8</td>
    <td>3*0.835</td>
    <td>499</td>
    <td>167</td>
    <td>IN</td>
  </tr>
  <tr align="center">
    <td align="left">Person</td>
    <td align="left">Nicolas Cage</td>
    <td>0.8</td>
    <td>3*0.710</td>
    <td>461</td>
    <td>167</td>
    <td>UN</td>
  </tr>
  <tr align="center">
    <td align="left">Human</td>
    <td align="left">3D render in the style of Pixar</td>
    <td>0.8</td>
    <td>3*0.667</td>
    <td>446</td>
    <td>167</td>
    <td>UN</td>
  </tr>
  <tr align="center">
    <td align="left">Human</td>
    <td align="left">Neanderthal</td>
    <td>1.2</td>
    <td>3*0.802</td>
    <td>490</td>
    <td>167</td>
    <td>UN</td>
  </tr>
  <tr align="center">
    <td align="left">photo</td>
    <td align="left">Painting in Modigliani style</td>
    <td>0.8</td>
    <td>3*0.565</td>
    <td>403</td>
    <td>167</td>
    <td>UN</td>
  </tr>
  <tr align="center">
    <td align="left">photo</td>
    <td align="left">self-portrait by Frida Kahlo</td>
    <td>0.8</td>
    <td>3*0.443</td>
    <td>321</td>
    <td>167</td>
    <td>UN</td>
  </tr>
  <tr align="left">
    <td colspan=7><b>Table 1.</b> Hyperparameter settings of reproducibility experiments. The "domain" column corresponds<br>to the attribute being in-domain (IN) or unseen-domain (UN).</td>
  </tr>
</table>

Figure 6 shows that the results obtained in the original Asyrp paper and presented in \[8, Figure 4\] can be successfully reproduced and that editing in the h-space results in high performance image generation for in-domain attributes. Nevertheless, we must stress that the methodology does not necessarily isolate attribute changes and particular shifts may also result in other unintended shifts. To give an example, edits in the "woman" and "young" directions appear heavily entangled for the first image in Figure 6.

| ![In-domain](figures/reproduction/in_1.0.png) | 
|:-:| 
| **Figure 6.** Editing results of Asyrp on CelebA-HQ for in-domain attributes. |

Figures 7 and 8 depict the results of our reproducibility experiment focused on unseen-domain attributes (i.e., attributes that have not been observed in the training data) originally presented in \[8, Figure 5\]. In Figure 7, we use the full $\Delta h_t$ as done by the authors. In Figure 8, we reduce the editing strength by taking $0.5 \Delta h_t$. We observe that for unseen-domain attributes, reduction of the editing strength can nicely reduce invasiveness of the method and produce more sound results.

| ![Unseen-domain](figures/reproduction/unseen_1.0.png) | 
|:-:| 
| **Figure 7.** Editing results of Asyrp on CelebA-HQ for unseen-domain attributes. |

| ![Unseen-domain](figures/reproduction/unseen_0.5.png) | 
|:-:| 
| **Figure 8.** Editing results of Asyrp on CelebA-HQ for unseen-domain attributes with $0.5 \Delta h_t$. |

To quantitatively appreciate the performance of the Asyrp model, we reproduce the evaluation they conducted and compute the Directional CLIP score for the same three in-domain attributes (smiling, sad, tanned) and two unseen-domain attributes (Pixar, Neanderthal) on a set of 100 images per attribute from the CelebA-HQ dataset. The available repository does not provide code for implementing neither of the evaluation metrics, which leads to also not knowing which 100 images from the dataset were considered when computing the scores. We took the first 100 images and the comparative results can be seen in Table 1. We did not implement the segmentation consistency score, as we showed in the Evaluation Diffusion Models section that it has shortcomings, but we computed the FID score that is more meaningful in the case of image editing.

<table align="center">
  <tr align="center">
      <th align="left">Metric</th>
      <th>Smiling (IN)</th>
      <th>Sad (IN)</th>
      <th>Tanned (IN)</th>
      <th>Pixar (UN)</th>
      <th>Neanderthal (UN)</th>
  </tr>
  <tr align="center">
    <td align="left">Original $S_{dir}$</td>
    <td>0.921</td>
    <td>0.964</td>
    <td>0.991</td>
    <td>0.956</td>
    <td>0.805</td>
  </tr>
  <tr align="center">
    <td align="left">Reproduced $S_{dir}$</td>
    <td>0.989</td>
    <td>1.003</td>
    <td>1.003</td>
    <td>0.987</td>
    <td>0.977</td>
  </tr>
  <tr align="center">
    <td align="left">Alt. metric</td>
    <td>0.966</td>
    <td>0.966</td>
    <td>0.962</td>
    <td>0.958</td>
    <td>0.953</td>
  </tr>
  <tr align="center">
    <td colspan=6><b>Table 2.</b> Asyrp's directional CLIP score for in-domain (I) and unseen-domain (U) attributes.</td>
  </tr>
</table>

<table align="center">
    <tr align="center">
      <th align="left">Metric</th>
      <th>Smiling (IN)</th>
      <th>Sad (IN)</th>
      <th>Tanned (IN)</th>
      <th>Pixar (UN)</th>
      <th>Neanderthal (UN)</th>
  </tr>
  <tr align="center"><td align="left">$FID(\mathbf{x}_{orig}, \mathbf{x}_{recon})$</td>
      <td>96.11</td>
      <td>96.11</td>
      <td>96.11</td>
      <td>96.11</td>
      <td>96.11</td>
  </tr>
  <tr align="center">
    <td align="left">$FID(\mathbf{x}_{orig}, \mathbf{x}_{edit})$</td>
    <td>80.42</td>
    <td>82.66</td>
    <td>92.17</td>
    <td>111.70</td>
    <td>93.84</td>
  </tr>
  <tr align="center">
    <td align="left">$FID(\mathbf{x}_{recon}, \mathbf{x}_{edit})$</td>
    <td>56.02</td>
    <td>50.42</td>
    <td>66.21</td>
    <td>80.61</td>
    <td>85.41</td>
  </tr>
  <tr align="center">
    <td colspan=6><b>Table 3.</b> Asyrp's $FID$ score for in-domain (I) and unseen-domain (U) attributes.</td>
  </tr>
</table>

We also conducted reproducibility experiments on the linearity and consistency across timesteps of the model. The results can be seen in Figures 8 and 9.

| ![Linearity](figures/linearity.png) | 
|:-:| 
| **Figure 8.** **TO-DO:** Linearity of h-space. |

| ![Linear combinations](figures/combination.png) | 
|:-:| 
| **Figure 9.** **TO-DO:** Linear combination of attributes. |

## Ablation study
While the reproduction results show that the general method works well, we set out to investigate further improvements by running an abaltion study. As previously mentioned in the [fourth](#architecture) section adjustments to the model architecture could provide further gains in performance in terms of the clip similairty, flexibility and transferability. In this section, we conduct several ablations in order to gain a deeper understanding of the asyrp method, aiming to identify its limitations and explore potential improvements.

### Hyperparameter dependency



### Ablations of the architecture
The original training performs well as we can see from the previous section, but is not further explored. Adjustments to this architecture could provide further gains in performance in terms of the clip similairty, flexibility and transferability.
The original model as seen in figure 4 can be broken down in multiple submodules. A input processing module, a temporal embedding module and an output processing module. In this section we will look more closely at these modules and propose several adjustments, which we compare to the original implementation.

#### pre- and postprocessing modules
The input and output of the module is an embedding of size w x h x c, in the celebAHQ dataset these take on the values 8 x 8 x 512. Any architecture we might want to use to processing this embedding must thus take in and return an output of that shape.
In the original architecture 1x1 convolutions are used to exchange information between channels of the embedding. We propose to use a transformer based architecture instead to exchange information between the elements of the embedding more effectively. To use a transformer we need to interpret the embedding as a sequence of length $n$ of $d$-dimensional tokens. 

We propose two ways of reinterpreting the data to get these sequences. We either interpret the channel dimensions of the image as the token dimension, resulting in a sequence length of $n=64$ with tokens of dimensions $d=512$ (pixel), or we swap these and get a sequence length $n=512$ with tokens of dimensions $d=64$ (channel). As both of these modules return an output of the same size as the input, we can also combine these two interpretations and apply them in serial.

We use a single transformer layer with a linear layer of dimension 2048 and use it to replace the convolutional layers in the pre- and postprocessing modules. We apply four variants, pixel, channel, pixel-channel & channel-pixel and train them for four epochs and calculate clip loss and FID. We report the results in table 4. We then pick the architecture with the lowest clip_loss, pixel-channel, and train it with 1,2,4 & heads.

#### Temporal embedding module
The temporal information about the denoising step is integrated into the original model by first linearly projecting the timestep embedding and then adding it to the embedding that was processed by the input module. In this section we investigate the integration of the temporal embedding by changing this addition to a multiplication, additionally we also test integrating the temporal embedding using an adjusted adaptive group norm.

#### activation function
A nonlinearity is applied after the group norm just before the output module


### Transfer-Learning between attributes
During training we often observed that the model first has to learn how to reconstruct the original image, effectively ignoring the added asyrp architecture, before it learns to edit the image through the clip directional loss. 
We therefore hypothesize that using pretrained weights from a different attribute than the target attribute should speed up training. We perform transfer learning from the 

### results

### Bias in editing directions
The editing directions found through the asyrp algorithm depend on the knowledge of attributes contained in CLIP. We observe in the output results that these editing directions are often highly biased. Individuals frequently change gender, skin color and eye color when edited with a direction that does not explicitely contain that change. For example, the Pixar editing direction changes the eyecolor of the source images to blue and often changes dark skin to white skin. This effect likely results from the model not being able to disentangle these concepts and has an impact on how useful these directions are in various image editing contexts. We have included some examples of these biased editing directions below.

### Transferability

## Further Research: Latent Diffusion Models:
Lastly in this blog post we set out to investigate whether Asyrp can also be applied on top of a latent diffusion model. Since LDMs currently represent the state-of-the-art in image generation \[16\], it is reasonable to find out if modifications in the h-space lead to meaningful attribute edits in the original images. Conveniently DDIM, the algorithm on which Asyrp was build, is also the algorithm behind LDMs. However, the diffusion process runs in the latent space instead of the pixel space. A sperate VQ-VAE  is trained \[19\], where the encoder $\mathcal{E}$ is used to compress the image $x_0$ to a smaller latent vector $z_0$ and the decoder $\mathcal{D}$ is used to reconstruct the image $\hat{x}_0$ from the computed latent vector $\hat{z}_0$. All the remaining steps are as described in the [second](#recap) and [third](#discover) section, but replacing $x$ by $z$. This leads to training a neural network $\epsilon\_\theta \left( z_t, t \right)$ and optimizing it with the loss in Equation 15. Furthermore, steps in the reverse process can be sampled with Equation 16.

$$L_{L D M} := \mathbb{E}\_{ \mathcal{E}(x), \epsilon \sim \mathcal{N}(0, 1), t } \left\[ \left\| \epsilon - \epsilon\_\theta \left( z_t, t \right) \right\|_2^2 \right\] \qquad \qquad \text{(Equation 15)}$$

$$z_{t-1} = \sqrt{\alpha\_{t-1}} \mathbf{P}\_t \left( \epsilon\_\theta \left( z_t, t \mid \Delta h_t \right) \right) + \mathbf{D}\_t \left( \epsilon\_\theta \left( z_t, t \right) \right) + \sigma_t v_t \qquad \qquad \text{(Equation 16)}$$

However, to calculate the directional CLIP loss both the reference and the generated image are needed, but the whole point of LDMs is that you do not calculate those every step. One aproach to still use the Asyrp algorithm could be to retrain CLIP for LDM latents instead of images, but this is beyond our scope. Therefor we investigated another aproach in which the images are computed from the latents by running the decoder $\mathcal{D}$ on $z_t$ at every time-step. Initially we questioned whether this approach would be fruitful as the VQ-VAE is only trained to reconstruct real images and not images perturbed by different levels of noise. In GIF 1 the results can be seen of running $\mathcal{D}$ on $z_t$ of a LDM at every time step. While this is no conclusive result, it does seem to hint that this approach would be feasible. 

| ![Linear combinations](src/lib/latent-diffusion/clip_loss_test/figures/output.gif) | 
|:-:| 
| **GIF 1.** Running the VQ-VAE decoder on the latent at every time step  |

That being said this section is called future research for a reason. Sadly the original code-base was not very modular and this made applying Asyrp to another DM or LDM not feasible within the scope of this project. Asyrp was build directly into a random DM code-base and thus applying it to a LDM would mean starting from scratch in a LDM code-base. Furthermore running the decoder on the latent and accessing the bottleneck feature map at every step meant that we had to edit low level code of large code-bases. Therefor eventually we decided to keep this as future research.

## Conclusion 

## Bibliography

[1] Jooyoung Choi, Sungwon Kim, Yonghyun Jeong, Youngjune Gwon, and Sungroh Yoon. [ILVR: Conditioning Method for Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2108.02938). In: CVF International Conference on Computer Vision (ICCV). 2021, pp. 14347–14356.

[2] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. “Arcface: Additive angular margin loss for deep face recognition”. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019, pp. 4690–4699.

[3] Prafulla Dhariwal and Alexander Nichol. “Diffusion models beat gans on image synthesis”. In: Advances in Neural Information Processing Systems 34 (2021), pp. 8780–8794.

[4] Rinon Gal, Or Patashnik, Haggai Maron, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. “StyleGAN-NADA: CLIP-guided domain adaptation of image generators”. In: ACM Transactions on Graphics (TOG) 41.4 (2022), pp. 1–13.[18]

[5] Jonathan Ho, Ajay Jain, and Pieter Abbeel. “Denoising diffusion probabilistic models”. In: Advances in Neural Information Processing Systems 33 (2020), pp. 6840–6851.

[6] Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. “Progressive growing of gans for improved quality, stability, and variation”. In: arXiv preprint arXiv:1710.10196 (2017).

[7] Gwanghyun Kim and Jong Chul Ye. “Diffusionclip: Text-guided image manipulation using diffusion models”. In: (2021).

[8] Mingi Kwon, Jaeseok Jeong, and Youngjung Uh. “Diffusion models already have a semantic latent space”. In: arXiv preprint arXiv:2210.10960 (2022).

[9] Xihui Liu, Dong Huk Park, Samaneh Azadi, Gong Zhang, Arman Chopikyan, Yuxiao Hu, Humphrey Shi, Anna Rohrbach, and Trevor Darrell. “More control for free! image synthesis with semantic diffusion guidance”. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2023, pp. 289–299.

[10] Chenlin Meng, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. “Sdedit: Image synthesis and editing with stochastic differential equations”. In: arXiv preprint arXiv:2108.01073 (2021).

[11] Alexander Quinn Nichol and Prafulla Dhariwal. “Improved denoising diffusion probabilistic models”. In: International Conference on Machine Learning. PMLR. 2021, pp. 8162–8171.

[12] Yong-Hyun Park, Mingi Kwon, Junghyo Jo, and Youngjung Uh. “Unsupervised Discovery of Semantic Latent Directions in Diffusion Models”. In: arXiv preprint arXiv:2302.12469 (2023).

[13] Or Patashnik, Zongze Wu, Eli Shechtman, Daniel Cohen-Or, and Dani Lischinski. “Styleclip: Text-driven manipulation of stylegan imagery”. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021, pp. 2085–2094.

[14] Konpat Preechakul, Nattanat Chatthee, Suttisak Wizadwongsa, and Supasorn Suwajanakorn. “Diffusion autoencoders: Toward a meaningful and decodable representation”. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022, pp. 10619-10629.

[15] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. “Learning transferable visual models from natural language supervision”. In: International conference on machine learning. PMLR. 2021, pp. 8748–8763.

[16] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj ̈orn Ommer. High-Resolution Image Synthesis with Latent Diffusion Models. 2021. arXiv: 2112.10752 [cs.CV].

[17] Jiaming Song, Chenlin Meng, and Stefano Ermon. “Denoising diffusion implicit models”. In: arXiv preprint arXiv:2010.02502 (2020).

[18] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens and Z. Wojna. [Rethinking the Inception Architecture for Computer Vision](https://ieeexplore.ieee.org/document/7780677). 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, USA, 2016, pp. 2818-2826, doi: 10.1109/CVPR.2016.308.

[19] Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu. "Neural Discrete Representation Learning". Advances in neural information processing systems 30 (2017). 
