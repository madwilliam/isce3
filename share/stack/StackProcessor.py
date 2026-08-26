from nisar.workflows.rubbersheet import run_rubbersheet_with_interpolation
import os
from pathlib import Path
import shutil
import pickle

from coarse_alignment import align_secondary, extract_ref_topo
from invert_offsets import invert_offsets_isce3
from nisar.workflows.helpers import copy_raster
from offset_util import (create_minimal_rifg_for_rubbersheet,
                         dense_offset_coregistered)
from ScanList import ScanList
from stack_utils import resample, relative_symlink_contents, set_nested, load_config

pol_freq = {'A': ['HH']}
runcfg = load_config()
cfg = runcfg.cfg["runconfig"]["groups"]
dense_cfg = cfg["processing"]["dense_offsets"]
rdr2geo_cfg = cfg['processing']['rdr2geo']
rubbersheet_cfg = cfg['processing']['rubbersheet']


class StackProcessor(ScanList):
    def __init__(
            self,
            slc_folder,
            out_folder,
            dem_file,
            pair_level=2,
            freq='A',
            pol='HH',
            overwrite=False,
            debug=False):
        self.slc_folder = Path(slc_folder)
        self.out_folder = Path(out_folder)
        self.config_save = self.out_folder / 'processor.pkl'
        self.dem_file = Path(dem_file)
        self.overwrite = overwrite
        self.freq = freq
        self.pol = pol
        self.debug = debug
        h5_files = [str(i) for i in self.slc_folder.glob('*1.h5')]
        super().__init__(h5_files)
        self.pol_freq = pol_freq
        self.pair_level = pair_level
        self.init_pairs()
        if not os.path.exists(self.config_save) or self.overwrite:
            pickle.dump(self, open(self.config_save, 'wb'))

    def init_pairs(self):
        self.dense_pairs = {}
        for path, scans in self.scan_by_id.items():
            nscans = len(scans)
            path_pairs = []
            for i in range(self.pair_level):
                if nscans - i > 1:
                    for j in range(nscans - i - 1):
                        path_pairs.append((scans[j], scans[j + i + 1]))
            self.dense_pairs[path] = path_pairs

    def iter_path_dense_pairs(self, path):
        yield from self.dense_pairs[path]

    def coarse_register_scans(self, rdr2geo_cfg=rdr2geo_cfg):
        for path in self.scan_by_id:
            main_ref = self.ref_scans[path]
            path_folder = self.out_folder / Path(path)
            stack_folder = path_folder / Path('stack')
            ref_geom_folder = path_folder / Path('ref_geom')
            coarse_offset_folder = path_folder / Path('coarse_offsets')
            os.makedirs(path_folder, exist_ok=True)
            os.makedirs(stack_folder, exist_ok=True)
            os.makedirs(coarse_offset_folder, exist_ok=True)
            reference_slc = stack_folder / Path(f'{main_ref.date_str}.slc')
            if not os.path.exists(ref_geom_folder) or self.overwrite:
                os.makedirs(ref_geom_folder, exist_ok=True)
                extract_ref_topo(
                    main_ref.path_str,
                    self.dem_file,
                    rdr2geo_cfg,
                    ref_geom_folder)
            if not os.path.exists(reference_slc) or self.overwrite:
                copy_raster(
                    main_ref.path_str,
                    self.freq,       # frequency
                    self.pol,      # polarization
                    1024,      # lines per block
                    reference_slc,
                    file_type="ENVI",
                )
            pairs = {}
            for ref, sec in self.iter_path_ref_sec(path):
                coarse_offset_folderi = coarse_offset_folder / \
                    Path(sec.date_str)
                val = {}
                val['azimuth'] = coarse_offset_folderi / \
                    Path(f'geo2rdr/freq{self.freq}/azimuth.off')
                val['range'] = coarse_offset_folderi / \
                    Path(f'geo2rdr/freq{self.freq}/range.off')
                pairs[(ref.date_str, sec.date_str)] = val
                if not os.path.exists(coarse_offset_folderi) or self.overwrite:
                    os.makedirs(coarse_offset_folderi, exist_ok=True)
                    align_secondary(
                        sec.path_str,
                        self.dem_file,
                        ref_geom_folder,
                        coarse_offset_folderi)

                sec_slc = stack_folder / Path(f'{sec.date_str}.slc')
                if not os.path.exists(sec_slc) or self.overwrite:
                    print(f'calculating coarse reg for {path} {sec.date_str}')
                    offset_dir = coarse_offset_folderi / "geo2rdr" / "freqA"
                    resample(
                        stack_folder,
                        offset_dir,
                        main_ref.path_str,
                        sec.path_str,
                        sec.date_str)

    def dense_offset_pairs(self, dense_cfg):
        for path in self.scan_by_id:
            for ref, sec in self.iter_path_dense_pairs(path):
                path_folder = self.out_folder / Path(path)
                dense_folder = path_folder / Path('dense_pairwise')
                stack_folder = path_folder / Path('stack')
                os.makedirs(dense_folder, exist_ok=True)
                starting_pair = f'{ref.date_str}_{sec.date_str}'
                out_dir = dense_folder / Path(starting_pair)
                if not os.path.exists(out_dir) or self.overwrite:
                    print(f'calculating dense offset {path} {starting_pair}')
                    os.makedirs(out_dir, exist_ok=True)
                    reference_slc = stack_folder / Path(f'{ref.date_str}.slc')
                    secondary_slc = stack_folder / Path(f"{sec.date_str}.slc")
                    dense_offset_coregistered(
                        reference_slc,
                        secondary_slc,
                        out_dir,
                        dense_cfg=dense_cfg,
                        gpu_id=0,
                    )

    def invert_dense_offset_pairs(self):
        for path in self.scan_by_id:
            pairs = {}
            path_folder = self.out_folder / Path(path)
            dense_folder = path_folder / Path('dense_pairwise')
            inverted_offsets_folder = path_folder / Path('inverted_offsets')
            if not os.path.exists(inverted_offsets_folder) or self.debug:
                os.makedirs(inverted_offsets_folder, exist_ok=True)
                for ref, sec in self.iter_path_dense_pairs(path):
                    starting_pair = f'{ref.date_str}_{sec.date_str}'
                    out_dir = dense_folder / Path(starting_pair)
                    pairs[(ref.date_str, sec.date_str)
                          ] = out_dir / 'dense_offsets'
                if len(pairs) > 0:
                    outputs = invert_offsets_isce3(
                        pair_offsets=pairs,
                        output_dir=str(inverted_offsets_folder),
                        reference_date=self.ref_scans[path].date_str,
                    )

    def rubbersheet(self, rubbersheet_cfg=rubbersheet_cfg):
        for path in self.scan_by_id:
            path_folder = self.out_folder / Path(path)
            coarse_offset_folder = path_folder / Path('coarse_offsets')
            inverted_offsets_folder = path_folder / Path('inverted_offsets')
            dense_folder = path_folder / Path('dense_pairwise')
            rubber_folder = path_folder / Path('rubber')
            h5_stack_folder = path_folder / Path('h5_stack')
            os.makedirs(h5_stack_folder, exist_ok=True)
            os.makedirs(rubber_folder, exist_ok=True)
            main_ref = self.ref_scans[path]
            for ref, sec in self.iter_path_ref_sec(path):
                coarse_offset_folderi = coarse_offset_folder / \
                    Path(sec.date_str)
                out_dir = h5_stack_folder / sec.date_str
                rubberi = rubber_folder / sec.date_str
                if not os.path.exists(rubberi) or self.overwrite:
                    print(f'calculating rubbersheet {path} {sec.date_str}')
                    cfg['processing']['rubbersheet'] = rubbersheet_cfg
                    inverted_offset_pathi = inverted_offsets_folder / sec.date_str
                    starting_pair = f'{ref.date_str}_{sec.date_str}'
                    densei = dense_folder / Path(starting_pair)
                    link_to = path_folder / Path('dense_offsets/freqA/HH')
                    relative_symlink_contents(densei, link_to)
                    relative_symlink_contents(inverted_offset_pathi, link_to)
                    cfg['dynamic_ancillary_file_group']['dem_file'] = str(
                        self.dem_file)
                    set_nested(
                        cfg, [
                            'input_file_group', 'reference_rslc_file'], main_ref.path_str)
                    set_nested(
                        cfg, [
                            'product_path_group', 'sas_output_file'], out_dir)
                    set_nested(
                        cfg, [
                            'product_path_group', 'scratch_path'], path_folder)
                    cfg['processing']['rubbersheet']['geo2rdr_offsets_path'] = coarse_offset_folderi
                    cfg['processing']['rubbersheet']['dense_offsets_path'] = path_folder
                    cfg['processing']['input_subset']['list_of_frequencies'] = {
                        self.freq: [self.pol]}
                    create_minimal_rifg_for_rubbersheet(
                        out_dir, cfg)
                    run_rubbersheet_with_interpolation(
                        cfg, out_dir)
                    shutil.move(
                        path_folder /
                        Path('rubbersheet_offsets'),
                        rubberi)

    def resample_slcs(self):
        for path in self.scan_by_id:
            path_folder = self.out_folder / Path(path)
            merged_folder = path_folder / Path('merged')
            rubber_folder = path_folder / Path('rubber')
            os.makedirs(merged_folder, exist_ok=True)
            main_ref = self.ref_scans[path]
            ref_slc_folder = merged_folder / main_ref.date_str
            reference_slc = ref_slc_folder / Path(f'{main_ref.date_str}.slc')
            if not os.path.exists(reference_slc) or self.overwrite:
                print(f'copying reference slc for {path}')
                os.makedirs(ref_slc_folder, exist_ok=True)
                copy_raster(
                    main_ref.path_str,
                    self.freq,       # frequency
                    self.pol,      # polarization
                    1024,      # lines per block
                    reference_slc,
                    file_type="ENVI",
                )
            for _, sec in self.iter_path_ref_sec(path):
                print(f'calculating final offset for {path} {sec.date_str}')
                rubberi = rubber_folder / sec.date_str / \
                    f'freq{self.freq}' / self.pol
                pairs_out = merged_folder / sec.date_str
                if not os.path.exists(pairs_out) or self.overwrite:
                    print(f'resampling {path} {sec.date_str}')
                    resample(
                        pairs_out,
                        rubberi,
                        main_ref.path_str,
                        sec.path_str,
                        out_tag=sec.date_str)
