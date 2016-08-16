# works on Linux Ubuntu 64 bit
mkdir ../data
mkdir ../data/benchmark
cd ../data/benchmark
wget http://www.semantic3d.net/data/point-clouds/training1/bildstein_station1_xyz_intensity_rgb.7z
wget http://www.semantic3d.net/data/point-clouds/testing2/MarketplaceFeldkirch_Station4_rgb_intensity-reduced.txt.7z
wget http://www.semantic3d.net/data/point-clouds/testing2/StGallenCathedral_station6_rgb_intensity-reduced.txt.7z
wget http://www.semantic3d.net/data/point-clouds/testing2/sg27_station10_rgb_intensity-reduced.txt.7z
wget http://www.semantic3d.net/data/point-clouds/testing2/sg28_Station2_rgb_intensity-reduced.txt.7z
wget http://www.semantic3d.net/data/point-clouds/training1/sg28_station4_intensity_rgb.7z
wget http://www.semantic3d.net/data/sem8_labels_training.7z
wget https://dl.dropboxusercontent.com/u/7069946/p7zip-binary-64bit.zip
unzip p7zip-binary-64bit.zip
chmod +x p7zip/7za
for arc in *.7z
do
  p7zip/7za e $arc
done
convert_to_train () {
  cut -d' ' -f1,2,3 $1.txt > temp.txt
  paste -d' ' temp.txt $1.labels > $1_train.txt
}
convert_to_test () {
  cut -d' ' -f1,2,3 $1.txt > $1_test.txt
}
convert_to_train sg28_station4_intensity_rgb
convert_to_train bildstein_station1_xyz_intensity_rgb
convert_to_test MarketplaceFeldkirch_Station4_rgb_intensity-reduced
convert_to_test StGallenCathedral_station6_rgb_intensity-reduced
convert_to_test sg27_station10_rgb_intensity-reduced
convert_to_test sg28_Station2_rgb_intensity-reduced
rm temp.txt
for file in *_test.txt; do sed -i "s/$/ 0/" $file; done
