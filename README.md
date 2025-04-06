## MOL3022 Bioinformatics

## Project created by:
- Trym Hamer Gudvangen  
- Nicolai Hollup Brand  
- Tomas Beranek  

## Web Application

This application is not hosted on a dedicated server and must be run locally on your computer. Follow the instructions below to clone the repository and set up the application.

![app-demo](/images/app-show.gif)

A more comprehensive guide can be found at:
https://youtu.be/5C_kukZYfKQ

### Clone Repository

#### Using HTTP:
```bash
git clone https://github.com/TrymNOHG/2prot-struct.git
```

#### Using SSH:
```bash
git clone git@github.com:TrymNOHG/2prot-struct.git
```

### Run the Web Application
To run and use the application see the README.md file in './app' folder. 

## Secondary Structure Prediction

The main purpose of this application and its associated Python scripts is to predict the Q8 secondary structure of protein sequences based on their amino acid sequences. This is achieved by first tokenizing and embedding each individual amino acid and then training a machine learning model to predict the corresponding secondary structure.

### Dataset
The dataset used for this project originates from the PS4 dataset created by Omar Peracha: [PS4 Dataset](https://github.com/omarperacha/ps4-dataset/tree/main)

All relevant data can be found in `data.csv`. Additionally, the FASTA entries from PCB are included in `data.fasta`, though they are not necessary for this project. 

To benchmark the performance of our secondary structure prediction model, we also include commonly used datasets such as **CB513** and **TS115**.

#### Dataset Format
The dataset is structured as follows:

| Column      | Description |
|------------|-------------|
| **chain_id** | The corresponding ID for the protein sequence from the Protein Data Bank (PDB). The first four characters of the chain ID can be used to locate the structure on [PDB](https://www.rcsb.org/structure/) (e.g., `4TRT` corresponds to [4TRT](https://www.rcsb.org/structure/4TRT)). |
| **first_res** | The index of the first amino acid residue in the sequence. If `first_res` is 19, the sequence starts at residue 19 in the full protein FASTA sequence. |
| **input** | The amino acid sequence that will be analyzed for its secondary structure. |
| **dssp8** | The corresponding secondary structure annotations using DSSP classification. |

### DSSP Secondary Structure Classes
DSSP classifies secondary structures into eight different types:

- **G** – 3₁₀ helix  
- **H** – α-helix  
- **I** – π-helix  
- **E** – β-sheet  
- **B** – β-bridge  
- **T** – Helix turn  
- **S** – Bend  
- **C** – Coil   
- **P** – Polyproline II (PPII) helix (sometimes included in Q8 prediction for extended analysis)  

## Tokenization and Embedding
The input amino acid sequences are first tokenized using a [protTokenizer](https://huggingface.co/Rostlab/prot_bert),
which is a scientifically validated tokenization method.
Each individual amino acid is then embedded into a numerical representation
that captures relevant biomechanical and structural properties. 

The embeddings were created using a large language model (LLM)
based on the pre-trained [protBERT](https://huggingface.co/Rostlab/prot_bert) model.
This approach was chosen to save time and achieve the most accurate results possible.
The last hidden layer of the **protBERT** model generates the embeddings,
which are then used as input for the prediction model.

## Baseline Models  
To evaluate the effectiveness of our approach, we compare our prediction model against simple baseline models.
To test the accuracy and the benefits of utilizing embeddings,
the baseline models used either a sliding window approach or predicted based on the amino acid directly.  

The different approaches can be seen in the list below, together with the implemented baseline models:

- **MLP Window Model:** (Sliding Window of a given length, creating multiple input values. This way, the model predicts the secondary structure based on neighboring amino acids.)  
- **Naive Bayes Model:** (Predicts the secondary protein structures based on amino acid sequences. It learns the probability of a secondary structure given an amino acid and then predicts the most probable structure for each residue in a sequence.)  
- **Simple Window Model:** (Sliding Window of a given length, creating multiple input values. This way, the model predicts the secondary structure based on neighboring amino acids.)  
- **Stochastic Model:** (Predicts based on the whole sequence by learning the probability distribution of each DSSP8 state in the training data. It then generates a prediction by sampling from this distribution.)  

## Prediction Model  
The final prediction model is a machine learning model trained to predict the secondary structure of an amino acid in the protein sequence.  
This is done based on its amino acid embedding, which is processed through a neural network and outputs the prediction frequency distribution for each residue.  
The model is a convolutional neural network that uses LSTM layers for prediction.  
This way, more of the structure of the embedding is learned by the model, which should result in better accuracy.  
This will also be the model used in the web application for predicting the structure of the input sequence.  

## Running the Model  
Disclaimer: We suggest that most of the code here is run on IDUN, as
the GPU and RAM requirements are quite large.
To run the model and obtain predictions, execute the following scripts:  

### Install Dependencies  
Ensure all required dependencies are installed:  
```bash  
pip install -r requirements.txt  
```

### Generate Embeddings  
Run the following script to tokenize and embed the input sequences:  
```bash  
python make_embeddings.py  
```

### Run Baseline Models  
To evaluate the baseline models, run:  
```bash  
python run_model.py  
```

### Run Prediction Model  
Execute the following command to run the final prediction model:  
```bash  
python run_lstm.py  
```
