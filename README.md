# Detecting Anomalies in Financial Transactions

### Classification of Financial Anomalies

Derived from this observation we distinguish two classes of anomalous journal entries, namely **"global"** and **"local" anomalies** as illustrated in **Figure 2** below:

![image](https://user-images.githubusercontent.com/64821137/231603556-ac7f2d61-14f5-4bc9-804b-859059f4104f.png)

**Figure 2:** Illustrative example of global and local anomalies portrait in a feature space of the two transaction features "Posting Amount" (Feature 1) and "Posting Positions" (Feature 2).

***Global Anomalies***, are financial transactions that exhibit **unusual or rare individual attribute values**. These anomalies usually relate to highly skewed attributes e.g. seldom posting users, rarely used ledgers, or unusual posting times. 

Traditionally "red-flag" tests, performed by auditors during annual audits, are designed to capture those types of anomalies. However, such tests might result in a high volume of false positive alerts due to e.g. regular reverse postings, provisions and year-end adjustments usually associated with a low fraud risk.

***Local Anomalies***, are financial transactions that exhibit an **unusual or rare combination of attribute values** while the individual attribute values occur quite frequently e.g. unusual accounting records. 

This type of anomaly is significantly more difficult to detect since perpetrators intend to disguise their activities trying to imitate a regular behaviour. As a result, such anomalies usually pose a high fraud risk since they might correspond to e.g. misused user accounts, irregular combinations of general ledger accounts and posting keys that don't follow an usual activity pattern.

The objective of this lab is to walk you through a deep learning based methodology that can be used to detect of global and local anomalies in financial datasets. The proposed method is based on the following assumptions: 

>1. the majority of financial transactions recorded within an organizations’ ERP-system relate to regular day-to-day business activities and perpetrators need to deviate from the ”regular” in order to conduct fraud,
>2. such deviating behaviour will be recorded by a very limited number of financial transactions and their respective attribute values or combination of attribute values and we refer to such deviation as "anomaly".

Concluding from these assumptions we can learn a model of regular journal entries with minimal ”harm” caused by the potential anomalous ones.

In order to detect such anomalies, we will train deep autoencoder networks to learn a compressed but "lossy" model of regular transactions and their underlying posting pattern. Imposing a strong regularization onto the network hidden layers limits the networks' ability to memorize the characteristics of anomalous journal entries. Once the training process is completed, the network will be able to reconstruct regular journal entries, while failing to do so for the anomalous ones.

After completing the lab you should be familiar with:

>1. the basic concepts, intuitions and major building blocks of autoencoder neural networks,
>2. the techniques of pre-processing financial data in order to learn a model of its characteristics,
>3. the application of autoencoder neural networks to detect anomalies in large-scale financial data, and,
>4. the interpretation of the detection results of the networks as well as its reconstruction loss. 

Please note, that this lab is neither a complete nor comprehensive forensic data analysis approach or fraud examination strategy. However, the methodology and code provided in this lab can be modified or adapted to detect anomalous records in a variety of financial datasets. Subsequently, the detected records might serve as a starting point for a more detailed and substantive examination by auditors or compliance personnel.

### Initial Data and Attribute Assessment

The dataset was augmented and renamed the attributes to appear more similar to a real-world dataset that one usually observes in SAP-ERP systems as part of SAP's Finance and Cost controlling (FICO) module. 

The dataset contains a subset of in total 7 categorical and 2 numerical attributes available in the FICO BKPF (containing the posted journal entry headers) and BSEG (containing the posted journal entry segments) tables. Please, find below a list of the individual attributes as well as a brief description of their respective semantics:

>- `BELNR`: the accounting document number,
>- `BUKRS`: the company code,
>- `BSCHL`: the posting key,
>- `HKONT`: the posted general ledger account,
>- `PRCTR`: the posted profit center,
>- `WAERS`: the currency key,
>- `KTOSL`: the general ledger account key,
>- `DMBTR`: the amount in local currency,
>- `WRBTR`: the amount in document currency.

Let's also have a closer look into the top 10 rows of the dataset:

<p align="center">
  <img src="https://user-images.githubusercontent.com/64821137/231604111-a92a5690-7cb1-4f08-a860-63f0d445f84b.png" />
</p>

You may also have noticed the attribute `label` in the data. We will use this field throughout to evaluate the quality of our trained models. The field describes the true nature of each individual transaction of either being a **regular** transaction (denoted by `regular`) or an **anomaly** (denoted by `global` and `local`).

### Autoencoder Neural Networks (AENNs)

The objective of this section is to familiarize ourselves with the underlying idea and concepts of building a deep autoencoder neural network (AENN). We will cover the major building blocks and the specific network structure of AENNs as well as an exemplary implementation using the open source machine learning library PyTorch.

### Autoencoder Neural Network Architecture



AENNs or "Replicator Neural Networks" are a variant of general feed-forward neural networks that have been initially introduced by Hinton and Salakhutdinov in [6]. AENNs usually comprise a **symmetrical network architecture** as well as a central hidden layer, referred to as **"latent"** or **"coding" layer**, of lower dimensionality. The design is chosen intentionally since the training objective of an AENN is to reconstruct its input in a "self-supervised" manner. 

below illustrates a schematic view of an autoencoder neural network:

![image](https://user-images.githubusercontent.com/64821137/231604818-312778e7-a652-450f-82a1-ab242fb1a5cf.png)

**Figure:** Schematic view of an autoencoder network comprised of two non-linear mappings (fully connected feed forward neural networks) referred to as encoder $f_\theta: \mathbb{R}^{dx} \mapsto \mathbb{R}^{dz}$ and decoder $g_\theta: \mathbb{R}^{dz} \mapsto \mathbb{R}^{dx}$.

Furthermore, AENNs can be interpreted as "lossy" data **compression algorithms**. They are "lossy" in a sense that the reconstructed outputs will be degraded compared to the original inputs. The difference between the original input $x^i$ and its reconstruction $\hat{x}^i$ is referred to as **reconstruction error**. In general, AENNs encompass three major building blocks:


>   1. an encoding mapping function $f_\theta$, 
>   2. a decoding mapping function $g_\theta$, 
>   3. and a loss function $\mathcal{L_{\theta}}$.

Most commonly the encoder and the decoder mapping functions consist of **several layers of neurons followed by a non-linear function** and shared parameters $\theta$. The encoder mapping $f_\theta(\cdot)$ maps an input vector (e.g. an "one-hot" encoded transaction) $x^i$ to a compressed representation $z^i$ referred to as latent space $Z$. This hidden representation $z^i$ is then mapped back by the decoder $g_\theta(\cdot)$ to a re-constructed vector $\hat{x}^i$ of the original input space (e.g. the re-constructed encoded transaction). Formally, the nonlinear mappings of the encoder- and the decoder-function can be defined by:

<p align="center">
  $f_\theta(x^i) = s(Wx^i + b)$, and $g_\theta(z^i) = s′(W′z^i + d)$
</p>

where $s$ and $s′$ denote non-linear activations with model parameters $\theta = \{W, b, W', d\}$, $W \in \mathbb{R}^{d_x \times d_z}, W' \in \mathbb{R}^{d_z \times d_y}$ are weight matrices and $b \in \mathbb{R}^{dx}$, $d \in \mathbb{R}^{dz}$ are offset bias vectors.

### Autoencoder Neural Network Implementation

Some elements of the encoder network code below should be given particular attention:

>- `self.encoder_Lx`: defines the linear transformation of the layer applied to the incoming input: $Wx + b$.
>- `nn.init.xavier_uniform`: inits the layer weights using a uniform distribution. 
>- `self.encoder_Rx`: defines the non-linear transformation of the layer: $\sigma(\cdot)$.
>- `self.dropout`: randomly zeros some of the elements of the input tensor with probability $p$.

We use **"Leaky ReLUs"** as introduced by Xu et al. to avoid "dying" non-linearities and to speed up training convergence. Leaky ReLUs allow a small gradient even when a particular neuron is not active. In addition, we include the **"drop-out" probability**, which defines the probability rate for each neuron to be set to zero at a forward pass to prevent the network from overfitting. 

Initially, we set the dropout probability of each neuron to $p=0.0$ (0%), meaning that none of the neuron activiations will be set to zero, the default interpretation of the dropout hyperparameter is the probability of training a given node in a layer, where 1.0 means no dropout, and 0.0 means no outputs from the layer.

### Evaluating the Autoencoder Neural Network (AENN) Model

The visualization reveals that the pre-trained model is able to reconstruct the majority of regular journal entries, while failing to do so, for the anomalous ones. As a result, the model reconstruction error can be used to distinguish both "global" anomalies (orange) and "local" anomalies (green) from the regular journal entries (blue).

<p align="center">
  <img src="https://user-images.githubusercontent.com/64821137/231606709-1c1b4aa9-41f4-4e41-b494-a5b2e0ebfd1a.png" />
</p>

