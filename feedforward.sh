for TRAIN_SIZE in {100000..900000..200000}
do
  for VOCAB_SIZE in {1000..10000..2000}
  do
    TRAIN_SIZE=$TRAIN_SIZE VOCAB_SIZE=$VOCAB_SIZE msub -v feedforward.pbs
  done
done
