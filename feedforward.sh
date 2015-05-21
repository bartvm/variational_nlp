export FUEL_DATA_PATH=$PWD
for TRAIN_SIZE in {100000..700000..100000}
do
  for VOCAB_SIZE in {1000..9000..1000}
  do
    TRAIN_SIZE=$TRAIN_SIZE VOCAB_SIZE=$VOCAB_SIZE msub -V feedforward.pbs
  done
done
