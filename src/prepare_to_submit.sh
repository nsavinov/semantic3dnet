#!/usr/bin/env bash

cd ../data/benchmark
mv MarketplaceFeldkirch_Station4_rgb_intensity-reduced_test.txt_predictions.txt marketsquarefeldkirch4-reduced.labels
mv sg27_station10_rgb_intensity-reduced_test.txt_predictions.txt sg27_10-reduced.labels
mv sg28_Station2_rgb_intensity-reduced_test.txt_predictions.txt sg28_2-reduced.labels
mv StGallenCathedral_station6_rgb_intensity-reduced_test.txt_predictions.txt stgallencathedral6-reduced.labels
zip submit.zip marketsquarefeldkirch4-reduced.labels sg27_10-reduced.labels sg28_2-reduced.labels stgallencathedral6-reduced.labels
