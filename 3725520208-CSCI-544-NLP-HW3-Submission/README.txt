Execute the python file normally.

The Output files, greedy_dev.out and viterbi_dev.out are the outputs that are predicted on the development data.

And the output files greedy.out and viterbi.out are the predictions generated on test data.

Use the below command to Evaluate the model on the Development Data for respective algorithms.
For Greedy Decoding Algorithm: python eval.py -p greedy_dev.out -g ./data/dev
For Viterbi Decoding Algorithm: python eval.py -p viterbi_dev.out -g ./data/dev

