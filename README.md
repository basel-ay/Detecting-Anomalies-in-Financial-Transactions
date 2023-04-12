# Detecting Anomalies in Financial Transactions

The Association of Certified Fraud Examiners estimates in its Global Fraud Study 2016 that the typical organization loses 5% of its annual revenues due to fraud. According to Joseph T. Wells the term **"fraud"** refers to, 

>_"the abuse of one's occupation for personal enrichment through the deliberate misuse of an organization's resources or assets"_. 

A similar more recent study, conducted by the auditors of PwC, revealed that 30% of the study respondents experienced losses of between \$100,000 and \$5 million USD in the last 24 months. The study also showed that financial statement fraud caused by far the greatest median loss of the surveyed fraud schemes.

At the same time organizations accelerate the digitization and reconfiguration of business processes affecting in particular Accounting Information Systems (AIS) or more general Enterprise Resource Planning (ERP) systems.

![image](https://user-images.githubusercontent.com/64821137/231603383-3e6147a9-eb00-40f8-8783-de512c9c46b5.png)


**Figure 1:** Hierarchical view of an Accounting Information System (AIS) that records distinct layers of abstraction, namely (1) the business process information, (2) the accounting information as well as the (3) technical journal entry information in designated database tables.

Steadily, these systems collect vast quantities of electronic evidence at an almost atomic level. This holds in particular for the journal entries of an organization recorded in its general ledger and sub-ledger accounts. SAP, one of the most prominent ERP software providers, estimates that approx. 76% of the world's transaction revenue touches one of their systems.

The illustration in **Figure 1** depicts a hierarchical view of an Accounting Information System (AIS) recording process and journal entry information in designated database tables. In the context of fraud examinations, the data collected by such systems may contain valuable traces of a potential fraud scheme.

### Classification of Financial Anomalies

When conducting a detailed examination of real-world journal entries, usually recorded in large-scaled AIS or ERP systems, two prevalent characteristics can be observed:

> - specific transactions attributes exhibit **a high variety of distinct attribute values** e.g. customer information, posted sub-ledgers, amount information, and 
> - the transactions exhibit **strong dependencies between specific attribute values** e.g. between customer information and type of payment, posting type and general ledgers. 

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
