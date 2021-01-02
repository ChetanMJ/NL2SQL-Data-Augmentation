## Data-Augmentation-for-Natural-Language-to-SQL-conversion
Most efforts on the Text-to-SQL problem, especially in academia, have focused on the model of translating natural language to SQL. However, in practice of a real system, it is even more important to get training data of large amount, high quality, and different NL variants, which is one general and fundamental piece in Text-to-SQL modeling work. Data augmentation techniques help improve performance by generating data of more variants from existing data without expensive human labeling. Recently, the reverse task of Text-to-SQL semantic parsing, SQL-to-Text question generation has been explored on the WikiSQL dataset as an automatic data augmentation approach. In this project, we performed extensive experiments on SQL-to-Text question generation on WikiSQL and a more challenging dataset, Spider. We then experimented with data augmentation using the generated questions as additional training data to train state-of-the-art models on each dataset.

Follow below steps of execution:
1. Covert SQL to Graph using the either one of the folders: 
    a. Spider to Graph - converts sql to graph for spider dataset 
    b. SQL to Graph - converts sql to tree structure graph
2. Using the graph data generated from step 1, use it in folder Graph2Seq-Pytorch folder to train the model to generate text for the each sql query.
3. This new text-sql combinations can be combined with original data to create augmented data.
4. This augmented data can be found in augmented_data folder
5. Using this augmnted data evaluate the NL2SQL-BERT-master model.

Instruction for each step can be found in respective folders
