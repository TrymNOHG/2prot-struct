The dataset used for this project comes from the project PS4 by Omar Peracha: https://github.com/omarperacha/ps4-dataset/tree/main

All of the relevant data can be found in data.csv. Additionally, the FASTA entries from PCB are added in data.fasta, but will not be necessary for the project. Common datasets used to benchmark solutions for secondary structure prediction are cb513 and ts115. Therefore, these datasets have also been added.

The format of the dataset is as follows:

chain_id,first_res,input,dssp8

**chain_id** - the corresponding id for the protein sequence found in the Protein Data Bank (PDB): https://www.rcsb.org/structure/
- This can be done by entering the first four characters of the chain id (e.g. https://www.rcsb.org/structure/4TRT for the first protein)

**first_res** - the index of the starting residue signifying the first amino acid in the whole protein sequence. If you look at the second protein sequence, the first_res is 19. This means if you retrieve the FASTA for the whole protein sequence, you would need to start reading from residue 19 to start reading the first amino acid. When you actually retrieve the fasta, you get this as the first 26 characters: MAFARGGLAQTASQTTSSPVRVGLSV... While in the data.FASTA, you will only see from PVRGLSV...

**input** - the part of the protein sequence that will be analyzed for its structure.

**dssp8** - the annotations of the structure for the protein sequence provided using DSSP classification

DSSP has 8 different secondary structure classes:

G – 310 helix
H – α-helix
I – π-helix
E – β-sheet
B – β-bridge
T – helix turn
S – bend (high curvature)
C – coil (none of the above)