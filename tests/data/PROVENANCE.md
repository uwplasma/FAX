# Fixture provenance

The one manifest of every external reference in `tests/data` and `examples/data`. The table below is checked by
`tests/test_provenance.py` (set of files, byte size, SHA-256), so a fixture cannot change or appear without an entry. The sections after it give, per
source, the identifier, URL, license, model switches, units, normalization, resolution, generation command and the test that
pins it with its tolerance. Frequencies are `f = omega / 2 pi` in kHz unless stated. Python scripts in `tests/data` are
generation commands, not fixtures, and are not hashed.

| File | Bytes | SHA-256 | Source |
|---|---|---|---|
| `tests/data/boozmn_ctok.nc` | 574896 | `63d3005cb8796ac64698405ea485b422faf6fc1224bdc84411a7d8584ec43403` | CTOK |
| `tests/data/stellgap_ctok_continuum.npz` | 21569 | `ff131c776a595c4830eecf7bfbd72b483e305c1e857b9dfa859fc545c12c1f60` | CTOK |
| `tests/data/ae3d_ctok_modes.npz` | 35148 | `b6e54a1d79e73bf614b0663f4bf4ecbd1559d2918549051274c1982b43359155` | CTOK |
| `tests/data/ae3d_v31_ctok.npz` | 324628 | `b3a06d3ab198f6fa77870cec47ed1ee577ab309d15016df4bdd632fdb4debcb1` | AE3D v31 |
| `tests/data/ae3d_v31_instrumentation.patch` | 4714 | `cf0f8ac9488742bc6f60d38c42c0d6454a690b6a3d7a6c19961ee3f67a5fc5f3` | AE3D v31 |
| `tests/data/knyazev_qa.npz` | 669852 | `1de48876527bc2bbc4bc84cdd464762c72e7f0fc0eaeb21a0cc991c1de0dedfb` | Knyazev QA |
| `examples/data/input.ITPA_Konies2018` | 473 | `e5a46235b4352f0a62ab237b09d463c539b752cd2f4f7c220ddf6ac5d51f3ee4` | VMEC decks |
| `examples/data/input.LandremanPaul2021_QA_lowres` | 6533 | `2687ca87a5d4e587153c1f64617f25c30d741a48b966ac1cee17fb4632b49857` | VMEC decks |
| `examples/data/input.LandremanPaul2021_QH_reactorScale_lowres` | 9518 | `830431ca2395aa98a4587914414e9e99fbeca39fad9ffd752a91681f6c9fd9e6` | VMEC decks |

## `knyazev_qa.npz` (0.67 MB): precise QA, STELLGAP continuum and AE3D band

| Item | Value |
|---|---|
| Identifier | Knyazev et al., "On shear Alfvén wave-induced energetic ion transport in optimized stellarators", *J. Plasma Phys.* (2026), arXiv:2603.03118. Data archive "Data and figure-generating scripts for: ...", Zenodo concept DOI 10.5281/zenodo.21299907, version DOI 10.5281/zenodo.21299908 |
| URL | <https://zenodo.org/records/21299908>, file `Knyazev_JPP2026.zip` (1 127 897 428 bytes) |
| License | CC-BY-4.0 (Zenodo record metadata). Attribution: Knyazev et al. (2026) |
| SHA-256 of the archive | `615ceea03bc96fa5687728b3fe3e10c62c3d2594d4d4d02bdeeef26fbb1e76f4` (MD5 `7f65aebb55500e7bba026fd4fd91510b`, as listed by Zenodo) |
| Members read (SHA-256) | `simulations_fig4_fig5/QA/all_inputs/boozmn_betaQA.nc` `6a32726c66b57dc5d68e31bbf53f6f5833f1e7fb124404dadfe76873beb2b6d5`; `FIG2-STELLGAP_and_AE3D_for_QI_and_QA/inputs/QA/stellgap/alfven_spec{0,1,2,3}` `cd4ef4aa19669e1a123464eea1f55632bbd04a5b6beeae1709c5ae2f85e8d212`, `9f1aafaeb0660d5e5bd84862cb698eeb03d6d49cdad827058551d86113f039a2`, `c6e2add386c7dc8a374ade1dea07d968e016871e86491c26870c191d197c5274`, `a2946c52fad44a94e11cb2660139d1a1ec2e915c77a009f745887fcda3024385`; `.../QA/ae3d/egn_mode_asci.dat` `78086943957525a6641578ee86ac5b289788af571cd476ae35359637d7d09511` |
| Model switches | Precise QA, `nfp = 2`, finite-beta VMEC equilibrium (`ns = 51`, aspect 6, `p(0) = 7.1e5` Pa) transformed by `booz_xform` (`mboz = 96`, `nboz = 23`). AE3D with its kink term removed (archive README); no pressure-curvature term either, so FAX runs with `finite_beta=False`. STELLGAP: the 162 branches of the same harmonic list. Sine basis `Phi = sum Phi_mn sin(m theta - n zeta)` in AE3D |
| Units, normalization | Flat mass density 1.88e-6 kg/m^3: the archive does not contain the STELLGAP/AE3D decks, so the value is inferred (it reproduces the STELLGAP continuum to 3e-3 and agrees with the archived AE3D frequency scale; the same value is used in PR #9). STELLGAP `omega^2 = alpha / beta` (columns 2 and 4 of `alfven_spec`, kHz^2); AE3D eigenvalue `= f^2` in kHz^2 (negative for unstable); AE3D eigenvectors are potential amplitudes at the nodes with arbitrary scale. STELLGAP labels surface `k` as `k / (ns - 1)`, one cell (0.02) outward of the physical `(k - 1) / (ns - 1)`, as in CTOK |
| Resolution | 49 radial nodes `s = 0.02, ..., 0.98` (the full mesh), 162 harmonics `(m, n)`, `m <= 26`, `n` from -2 to 16 in steps of 2 (full-torus `n`, `nfp = 2`). Stored: Boozer harmonics above 1e-5 of the largest (198 of 4489, float32); 98 of 784 STELLGAP surfaces; all 7938 AE3D frequencies; the 20 radially extended gap modes in 26-35 kHz (selection rule in `examples/knyazev_qa.py`) on their 96 largest harmonics, and the archived global mode (30.3086 kHz) on all 162 |
| Generation command | `python examples/knyazev_qa.py` with `ARCHIVE` set and `REBUILD = True` (prints the hashes above) |
| Pinned by, tolerance | `tests/test_knyazev.py`: STELLGAP continuum mean < 4e-3 and max < 5e-3 relative (measured 3.0e-3, 4.3e-3); archived global mode within 2% (+0.9% at 24 harmonics); band count 10-26 (AE3D 20). Budget in `docs/validation.md` ("Independent 3-D reference") |

## CTOK tokamak: `boozmn_ctok.nc`, `stellgap_ctok_continuum.npz`, `ae3d_ctok_modes.npz`

| Item | Value |
|---|---|
| Identifier | Public CTOK test problems of STELLGAP (ORNL-Fusion/Stellgap commit `9be3aa9`, 2025-08-07) and AE3D (ORNL-Fusion/AE3D commit `383d327`, 2019-05-29): circular tokamak, aspect 3.33, `ns = 101`, `nfp = 1`, `mboz = 48`, `nboz = 1`, 143 harmonics |
| URL | <https://github.com/ORNL-Fusion/Stellgap>, <https://github.com/ORNL-Fusion/AE3D> |
| License | MIT, (c) 2016 ORNL-Fusion (both repositories) |
| Upstream hashes | `stellgap_test_problem.tar.gz` `d37a2e3654dce36baffff8958a9d33920e1feb3fdb66885187b525cfadc823bb`; `stellgap_test_problem_answers.tar.gz` `b8900c5fa4e2ee3eab6ca668d96465f8205e0b240d5917a9ac32029d6a2885ec` (member `alfven_post`); `ae3d_test_problem.tar.gz` `7d106e2482e8584bf12b07c614da2b405cfbc0122b4825f8116b732999dde8ce`; `ae3d_test_problem_answers.tar.gz` `e3d8a3f3b5e5bffb5e57d44750d0de36652ff691a810a0a24988d64a19efaa2a` (member `egn_mode_asci.dat`) |
| Model switches | `n = 2`, `m = 0..12`, deuterium 1e19 m^-3 (`rho = 3.34e-8` kg/m^3, flat); AE3D without pressure and `J_par` terms (`finite_beta=False`) |
| Units, normalization | kHz, `f = omega / 2 pi`. STELLGAP file: 296 surfaces `s = 0.02 ... 0.9869` and the 13 sorted branches in kHz (STELLGAP labels shifted one cell, see `test_stellgap_ctok_parity`). AE3D file: 7 eigenvalues in kHz (337.92, 343.65, 384.13, 685.59, 703.30, 738.23, 741.21) and their eigenvectors on the 99-point full mesh, 13 harmonics, float32, amplitude normalized to 1 at the peak (arbitrary scale) |
| Resolution | 99 radial nodes, 13 harmonics |
| Generation command | `boozmn_ctok.nc` is the packet file, byte for byte (identical in both packets). The two `.npz` are rebuilt bit for bit: unpack the `*_answers.tar.gz` packets and run `python tests/data/make_ctok_fixtures.py STELLGAP_ANSWERS AE3D_ANSWERS OUT_DIR` (the 7 modes are the 0-based eigenvalue indices listed in the script) |
| Pinned by, tolerance | `tests/test_continuum.py::test_stellgap_ctok_parity` (mean and p95 relative error 7e-3), `tests/test_eigen.py::test_ae3d_ctok_parity` (6 modes within 1.5%, overlap > 0.9; the TAE is the unresolved discrepancy of `docs/validation.md`) |

## `ae3d_v31_ctok.npz`, `ae3d_v31_instrumentation.patch`: pinned AE3D v31 blocks

| Item | Value |
|---|---|
| Identifier | AE3D `ae_solve_version31.f` at ORNL-Fusion/AE3D commit `383d327c53bf722026d7e432797ea37c50347b68` (SHA-256 `d4cfd7d7ffa8ff0bb4c3a5b944e601473710fb4d3a841552046ced06f51d207f`) plus `ae3d_v31_instrumentation.patch`, run on the CTOK problem of `ae3d_test_problem.tar.gz` (hash above) |
| License | MIT (AE3D); the patch is a derivative of it, same license |
| Model switches | `plasma.dat` `ion_density_0 = 1e19`, `ion_to_proton_mass = 2`, `ion_profile = 2`; `AE3D_SW=D` (blocks dumped) and `AE3D_SW=SD` (bending shear terms kept; J and P off) |
| Units, normalization | AE3D internal block coefficients; metric columns `rjacob bfield gsssup gttsup gzzsup gstsup gszsup gtzsup bfields bfieldth bfieldze` of `ae_metric.dat` (AE3D replaces its last point by linear extrapolation) |
| Resolution | `ns = 99`, 13 harmonics, `80 x 80` angles; blocks of surfaces 45 and 46 (`s = 0.45, 0.46`) in float64, the full bending matrix diagonal and upper blocks and the metric of every second angle in float32 |
| Generation command | `patch -p1 < ae3d_v31_instrumentation.patch`, build with gfortran 15.2.0 (`-O2 -ffixed-form -ffixed-line-length-none -fallow-argument-mismatch`, LAPACK Accelerate), run twice with `AE3D_SW=D` and `AE3D_SW=SD`, then `python tests/data/make_ae3d_v31_ctok.py RUN_D RUN_SD tests/data/ae3d_v31_ctok.npz`. The run directories (128 MB `ae_metric.dat`) are not stored |
| Pinned by, tolerance | `tests/test_eigen.py::test_ae3d_v31_blocks_and_energy_on_identical_inputs`: inertia and base blocks 1e-9 to 1e-5, symmetric `iota'` part 1e-4, antisymmetric part 1e-3 (opposite sign by a factor 2 in the AE3D form) |

## VMEC decks `examples/data/input.*`

| Item | Value |
|---|---|
| Identifier | `input.ITPA_Konies2018`: written for FAX from the ITPA TAE benchmark equilibrium of Könies et al., *Nucl. Fusion* **58**, 126027 (2018), doi:10.1088/1741-4326/aae4e6 (circular tokamak `R = 10 + cos t`, `Z = sin t`, pressure `AM`, iota `AI`); no external file. `input.LandremanPaul2021_QA_lowres`, `..._QH_reactorScale_lowres`: low-resolution decks of the precise-QA and QH configurations of Landreman and Paul, *Phys. Rev. Lett.* **128**, 035001 (2022), copied from the VMEX `examples/data` (the QH deck is identical; the QA deck differs only in the last `NITER_ARRAY` entry, 2000 instead of 1000, up to whitespace) |
| License | MIT (this repository and VMEX); the Landreman-Paul decks derive from the public simsopt test inputs (MIT) |
| Model switches | Fixed-boundary VMEC input: ITPA `LASYM = F`, `NFP = 1`, `MPOL = 8`, `NS_ARRAY = 16 51`; QA `NFP = 2`; QH `NFP = 4` (reactor scale) |
| Units, normalization | VMEC SI units (Pa, Wb, m) |
| Resolution | as in the decks: ITPA `ns = 51`; QA and QH low resolution |
| Generation command | none (inputs); solved by `vmex` through `fax.from_vmec_input` |
| Pinned by, tolerance | `tests/test_vmex.py` and the examples (`slow`, `scheduled`); ITPA: `test_konies_itpa_finite_beta_tae` (3.947e5 rad/s with `p`, CAS3D 4.01e5; tolerances in `docs/validation.md`) |
