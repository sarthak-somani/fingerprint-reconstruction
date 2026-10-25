# Evaluation (evaluation version 2)

`evaluate.py` is the evaluator; `run_eval.py --exp EXX --split test` runs it on an experiment's repairs, files the
run under `results/<exp>/` and appends a row to `results/LEADERBOARD.md`; `compare_experiments.py --split test`
tests every pair of experiments (McNemar / Wilcoxon, Holm-adjusted, equivalence margins) into `results/comparison/`.

Protocol: every image (clean, damaged, repaired, gallery) is MSU-normalised (texture part of the cartoon-texture
decomposition, sigma 2.5). Each probe is matched with MINDTCT + BOZORTH3 against its own clean slap print (genuine)
and 50 fixed impostor prints; tau = the score at which 1% of the clean probes' impostor comparisons are accepted.
Headline scores, with 95% bootstrap CIs over fingers: **net rescue** (rescued − broken) / N, the **false-match gate**
(upper CI of FMR repaired − FMR damaged < +0.5 points) and **ridge fidelity** (local correlation of band-passed
repaired and clean ridges inside the damage mask). Diagnostics: EER, minutiae quality, NFIQ 2, PSNR, MS-SSIM, LPIPS.
Only the summary tables are in git (`results/LEADERBOARD.md`, `leaderboard.csv`, `comparison/*/comparison.md`).

## Third-party tools (`third_party/`, git-ignored)

The script looks in `third_party/nbis/bin` and `third_party/NFIQ2/bin` first, then on `PATH` (`FPR_NBIS_BIN`
overrides the NBIS folder). MINDTCT and BOZORTH3 are required; without `nfiq2` the report omits NFIQ 2.
Tested on RHEL 8 with gcc 8.5, conda and cmake from pip; run from `evaluation/third_party/`:

```bash
export CMAKE_POLICY_VERSION_MINIMUM=3.5          # cmake 4 refuses NBIS/NFIQ2's old cmake_minimum_required
mkdir -p nbis_src nbis && (cd nbis_src && curl -LO https://nigos.nist.gov/nist/nbis/nbis_v5_0_0.zip && unzip -q nbis_v5_0_0.zip)
git clone --depth 1 --branch v2.3.0 --recurse-submodules --shallow-submodules https://github.com/usnistgov/NFIQ2.git NFIQ2
git clone --depth 1 https://github.com/richzhang/PerceptualSimilarity.git          # official LPIPS for evaluate.py
# NBIS 5.0.0 (mindtct, bozorth3)
(cd nbis_src/Rel_5.0.0 && ./setup.sh "$PWD/../../nbis" --without-X11 --64 && make config && make it && make install LIBNBIS=no)
# NFIQ 2 v2.3.0: libraries from conda-forge, no static libstdc++, rpath to the prefix (~20 min, builds OpenCV)
conda create -y -p "$PWD/nfiq2_deps" -c conda-forge --override-channels libpng libtiff libjpeg-turbo openjpeg sqlite zlib libdb
sed -i 's/ -static-libgcc -static-libstdc++ -Wl,-z,defs/ -Wl,-z,defs/; s/"${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++")/"${CMAKE_EXE_LINKER_FLAGS}")/' \
    NFIQ2/cmake/compiler.cmake NFIQ2/fingerjetfxose/FingerJetFXOSE/cmake/compiler.cmake
export CMAKE_PREFIX_PATH="$PWD/nfiq2_deps" LDFLAGS="-Wl,-rpath,$PWD/nfiq2_deps/lib -L$PWD/nfiq2_deps/lib"
(cd NFIQ2 && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_NFIQ2_CLI=ON && cmake --build build -j 8 \
    && cp -r build/install_staging/nfiq2 install && ln -sfn install/bin bin)
```
