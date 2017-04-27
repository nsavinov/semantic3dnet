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
wget -O p7zip_16.02_x86_linux_bin.tar.bz2 "https://downloads.sourceforge.net/project/p7zip/p7zip/16.02/p7zip_16.02_x86_linux_bin.tar.bz2?r=https%3A%2F%2Fsourceforge.net%2Fprojects%2Fp7zip%2Ffiles%2Fp7zip%2F16.02%2F&ts=1493289214&use_mirror=netcologne"
tar jxf p7zip_16.02_x86_linux_bin.tar.bz2
mkdir p7zip
mv p7zip_16.02/bin/7za p7zip
rm -rf p7zip_16.02_x86_linux_bin.tar.bz2 p7zip_16.02
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
