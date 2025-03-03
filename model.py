from download import download_file, conf_edit

def get_model_config(clean_model, chunk_size=None, overlap=None):
    """Returns model type, config path, and checkpoint path for a given model name, downloading files if needed."""
    model_type, config_path, start_check_point = "", "", ""

    if clean_model == 'VOCALS-InstVocHQ':
            model_type = 'mdx23c'
            config_path = 'ckpts/config_vocals_mdx23c.yaml'
            start_check_point = 'ckpts/model_vocals_mdx23c_sdr_10.17.ckpt'
            download_file('https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_vocals_mdx23c.yaml')
            download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_mdx23c_sdr_10.17.ckpt')

    elif clean_model == 'VOCALS-MelBand-Roformer (by KimberleyJSN)':
            model_type = 'mel_band_roformer'
            config_path = 'ckpts/config_vocals_mel_band_roformer_kj.yaml'
            start_check_point = 'ckpts/MelBandRoformer.ckpt'
            download_file('https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml')
            download_file('https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'VOCALS-BS-Roformer_1297 (by viperx)':
            model_type = 'bs_roformer'
            config_path = 'ckpts/model_bs_roformer_ep_317_sdr_12.9755.yaml'
            start_check_point = 'ckpts/model_bs_roformer_ep_317_sdr_12.9755.ckpt'
            download_file('https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml')
            download_file('https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'VOCALS-BS-Roformer_1296 (by viperx)':
            model_type = 'bs_roformer'
            config_path = 'ckpts/model_bs_roformer_ep_368_sdr_12.9628.yaml'
            start_check_point = 'ckpts/model_bs_roformer_ep_368_sdr_12.9628.ckpt'
            download_file('https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_368_sdr_12.9628.ckpt')
            download_file('https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_368_sdr_12.9628.yaml')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'VOCALS-BS-RoformerLargev1 (by unwa)':
            model_type = 'bs_roformer'
            config_path = 'ckpts/config_bsrofoL.yaml'
            start_check_point = 'ckpts/BS-Roformer_LargeV1.ckpt'
            download_file('https://huggingface.co/jarredou/unwa_bs_roformer/resolve/main/BS-Roformer_LargeV1.ckpt')
            download_file('https://huggingface.co/jarredou/unwa_bs_roformer/raw/main/config_bsrofoL.yaml')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'VOCALS-Mel-Roformer big beta 4 (by unwa)':
            model_type = 'mel_band_roformer'
            config_path = 'ckpts/config_melbandroformer_big_beta4.yaml'
            start_check_point = 'ckpts/melband_roformer_big_beta4.ckpt'
            download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/melband_roformer_big_beta4.ckpt')
            download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-big/raw/main/config_melbandroformer_big_beta4.yaml')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'VOCALS-Melband-Roformer BigBeta5e (by unwa)':
            model_type = 'mel_band_roformer'
            config_path = 'ckpts/big_beta5e.yaml'
            start_check_point = 'ckpts/big_beta5e.ckpt'
            download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta5e.ckpt')
            download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta5e.yaml')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'INST-Mel-Roformer v1 (by unwa)':
            model_type = 'mel_band_roformer'
            config_path = 'ckpts/config_melbandroformer_inst.yaml'
            start_check_point = 'ckpts/melband_roformer_inst_v1.ckpt'
            download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/melband_roformer_inst_v1.ckpt')
            download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/raw/main/config_melbandroformer_inst.yaml')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'INST-Mel-Roformer v2 (by unwa)':
            model_type = 'mel_band_roformer'
            config_path = 'ckpts/config_melbandroformer_inst_v2.yaml'
            start_check_point = 'ckpts/melband_roformer_inst_v2.ckpt'
            download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/melband_roformer_inst_v2.ckpt')
            download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/raw/main/config_melbandroformer_inst_v2.yaml')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'INST-VOC-Mel-Roformer a.k.a. duality (by unwa)':
            model_type = 'mel_band_roformer'
            config_path = 'ckpts/config_melbandroformer_instvoc_duality.yaml'
            start_check_point = 'ckpts/melband_roformer_instvoc_duality_v1.ckpt'
            download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/melband_roformer_instvoc_duality_v1.ckpt')
            download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/raw/main/config_melbandroformer_instvoc_duality.yaml')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'INST-VOC-Mel-Roformer a.k.a. duality v2 (by unwa)':
            model_type = 'mel_band_roformer'
            config_path = 'ckpts/config_melbandroformer_instvoc_duality.yaml'
            start_check_point = 'ckpts/melband_roformer_instvox_duality_v2.ckpt'
            download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/melband_roformer_instvox_duality_v2.ckpt')
            download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/raw/main/config_melbandroformer_instvoc_duality.yaml')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'KARAOKE-MelBand-Roformer (by aufr33 & viperx)':
            model_type = 'mel_band_roformer'
            config_path = 'ckpts/config_mel_band_roformer_karaoke.yaml'
            start_check_point = 'ckpts/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt'
            download_file('https://huggingface.co/jarredou/aufr33-viperx-karaoke-melroformer-model/resolve/main/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt')
            download_file('https://huggingface.co/jarredou/aufr33-viperx-karaoke-melroformer-model/resolve/main/config_mel_band_roformer_karaoke.yaml')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'OTHER-BS-Roformer_1053 (by viperx)':
            model_type = 'bs_roformer'
            config_path = 'ckpts/model_bs_roformer_ep_937_sdr_10.5309.yaml'
            start_check_point = 'ckpts/model_bs_roformer_ep_937_sdr_10.5309.ckpt'
            download_file('https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_937_sdr_10.5309.ckpt')
            download_file('https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_937_sdr_10.5309.yaml')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'CROWD-REMOVAL-MelBand-Roformer (by aufr33)':
            model_type = 'mel_band_roformer'
            config_path = 'ckpts/model_mel_band_roformer _crowd.yaml'
            start_check_point = 'ckpts/mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt'
            download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.4/mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt')
            download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.4/model_mel_band_roformer_crowd.yaml')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'VOCALS-VitLarge23 (by ZFTurbo)':
            model_type = 'segm_models'
            config_path = 'ckpts/config_vocals_segm_models.yaml'
            start_check_point = 'ckpts/model_vocals_segm_models_sdr_9.77.ckpt'
            download_file('https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/refs/heads/main/configs/config_vocals_segm_models.yaml')
            download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_segm_models_sdr_9.77.ckpt')

    elif clean_model == 'CINEMATIC-BandIt_Plus (by kwatcharasupat)':
            model_type = 'bandit'
            config_path = 'ckpts/config_dnr_bandit_bsrnn_multi_mus64.yaml'
            start_check_point = 'ckpts/model_bandit_plus_dnr_sdr_11.47.chpt'
            download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.3/config_dnr_bandit_bsrnn_multi_mus64.yaml')
            download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.3/model_bandit_plus_dnr_sdr_11.47.chpt')

    elif clean_model == 'DRUMSEP-MDX23C_DrumSep_6stem (by aufr33 & jarredou)':
            model_type = 'mdx23c'
            config_path = 'ckpts/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.yaml'
            start_check_point = 'ckpts/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt'
            download_file('https://github.com/jarredou/models/releases/download/aufr33-jarredou_MDX23C_DrumSep_model_v0.1/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt')
            download_file('https://github.com/jarredou/models/releases/download/aufr33-jarredou_MDX23C_DrumSep_model_v0.1/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.yaml')

    elif clean_model == '4STEMS-SCNet_MUSDB18 (by starrytong)':
            model_type = 'scnet'
            config_path = 'ckpts/config_musdb18_scnet.yaml'
            start_check_point = 'ckpts/scnet_checkpoint_musdb18.ckpt'
            download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.6/config_musdb18_scnet.yaml')
            download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.6/scnet_checkpoint_musdb18.ckpt')

    elif clean_model == 'DE-REVERB-MDX23C (by aufr33 & jarredou)':
            model_type = 'mdx23c'
            config_path = 'ckpts/config_dereverb_mdx23c.yaml'
            start_check_point = 'ckpts/dereverb_mdx23c_sdr_6.9096.ckpt'
            download_file('https://huggingface.co/jarredou/aufr33_jarredou_MDXv3_DeReverb/resolve/main/dereverb_mdx23c_sdr_6.9096.ckpt')
            download_file('https://huggingface.co/jarredou/aufr33_jarredou_MDXv3_DeReverb/resolve/main/config_dereverb_mdx23c.yaml')

    elif clean_model == 'DENOISE-MelBand-Roformer-1 (by aufr33)':
            model_type = 'mel_band_roformer'
            config_path = 'ckpts/model_mel_band_roformer_denoise.yaml'
            start_check_point = 'ckpts/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt'
            download_file('https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt')
            download_file('https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/model_mel_band_roformer_denoise.yaml')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'DENOISE-MelBand-Roformer-2 (by aufr33)':
            model_type = 'mel_band_roformer'
            config_path = 'ckpts/model_mel_band_roformer_denoise.yaml'
            start_check_point = 'ckpts/denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt'
            download_file('https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt')
            download_file('https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/model_mel_band_roformer_denoise.yaml')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'VOCALS-MelBand-Roformer Kim FT (by Unwa)':
            model_type = 'mel_band_roformer'
            config_path = 'ckpts/config_kimmel_unwa_ft.yaml'
            start_check_point = 'ckpts/kimmel_unwa_ft.ckpt'
            download_file('https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft.ckpt')
            download_file('https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'inst_v1e (by unwa)':
            model_type = 'mel_band_roformer'
            config_path = 'ckpts/config_melbandroformer_inst.yaml'
            start_check_point = 'ckpts/inst_v1e.ckpt'
            download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/inst_v1e.ckpt')
            download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/config_melbandroformer_inst.yaml')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'bleed_suppressor_v1 (by unwa)':
            model_type = 'mel_band_roformer'
            config_path = 'ckpts/config_bleed_suppressor_v1.yaml'
            start_check_point = 'ckpts/bleed_suppressor_v1.ckpt'
            download_file('https://huggingface.co/ASesYusuf1/MODELS/resolve/main/bleed_suppressor_v1.ckpt')
            download_file('https://huggingface.co/ASesYusuf1/MODELS/resolve/main/config_bleed_suppressor_v1.yaml')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'VOCALS-MelBand-Roformer (by Becruily)':
            model_type = 'mel_band_roformer'
            config_path = 'ckpts/config_instrumental_becruily.yaml'
            start_check_point = 'ckpts/mel_band_roformer_vocals_becruily.ckpt'
            download_file('https://huggingface.co/becruily/mel-band-roformer-vocals/resolve/main/config_vocals_becruily.yaml')
            download_file('https://huggingface.co/becruily/mel-band-roformer-vocals/resolve/main/mel_band_roformer_vocals_becruily.ckpt')
            conf_edit(config_path, chunk_size, overlap)
    
    elif clean_model == 'INST-MelBand-Roformer (by Becruily)':
            model_type = 'mel_band_roformer'
            config_path = 'ckpts/config_instrumental_becruily.yaml'
            start_check_point = 'ckpts/mel_band_roformer_instrumental_becruily.ckpt'
            download_file('https://huggingface.co/becruily/mel-band-roformer-instrumental/resolve/main/config_instrumental_becruily.yaml')
            download_file('https://huggingface.co/becruily/mel-band-roformer-instrumental/resolve/main/mel_band_roformer_instrumental_becruily.ckpt')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == '4STEMS-SCNet_XL_MUSDB18 (by ZFTurbo)':
            model_type = 'scnet'
            config_path = 'ckpts/config_musdb18_scnet_xl.yaml'
            start_check_point = 'ckpts/model_scnet_ep_54_sdr_9.8051.ckpt'
            download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.13/config_musdb18_scnet_xl.yaml')
            download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.13/model_scnet_ep_54_sdr_9.8051.ckpt')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == '4STEMS-SCNet_Large (by starrytong)':
            model_type = 'scnet'
            config_path = 'ckpts/config_musdb18_scnet_large_starrytong.yaml'
            start_check_point = 'ckpts/SCNet-large_starrytong_fixed.ckpt'
            download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.9/config_musdb18_scnet_large_starrytong.yaml')
            download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.9/SCNet-large_starrytong_fixed.ckpt')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == '4STEMS-BS-Roformer_MUSDB18 (by ZFTurbo)':
           model_type = 'bs_roformer'
           config_path = 'ckpts/config_bs_roformer_384_8_2_485100.yaml'
           start_check_point = 'ckpts/model_bs_roformer_ep_17_sdr_9.6568.ckpt'
           download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/config_bs_roformer_384_8_2_485100.yaml')
           download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/model_bs_roformer_ep_17_sdr_9.6568.ckpt')
           conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'DE-REVERB-MelBand-Roformer aggr./v2/19.1729 (by anvuew)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/dereverb_mel_band_roformer_anvuew.yaml'
          start_check_point = 'ckpts/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt'
          download_file('https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt')
          download_file('https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml')
          conf_edit(config_path, chunk_size, overlap)
          
    elif clean_model == 'DE-REVERB-Echo-MelBand-Roformer (by Sucial)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/config_dereverb-echo_mel_band_roformer.yaml'
          start_check_point = 'ckpts/dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt'
          download_file('https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt')
          download_file('https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/config_dereverb-echo_mel_band_roformer.yaml')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'dereverb_mel_band_roformer_less_aggressive_anvuew':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/dereverb_mel_band_roformer_anvuew.yaml'
          start_check_point = 'ckpts/dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt'
          download_file('https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml')
          download_file('https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'dereverb_mel_band_roformer_anvuew':
          model_type = 'mel_band_roformer'
          config_path = 'dereverb_mel_band_roformer_anvuew.yaml'
          start_check_point = 'ckpts/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt'
          download_file('https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml')
          download_file('https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt')
          conf_edit(config_path, chunk_size, overlap)  


    elif clean_model == 'inst_gabox (by Gabox)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/inst_gabox.yaml'
          start_check_point = 'ckpts/inst_gabox.ckpt'
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'inst_gaboxBV1 (by Gabox)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/inst_gabox.yaml'
          start_check_point = 'ckpts/inst_gaboxBv1.ckpt'
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxBv1.ckpt')
          conf_edit(config_path, chunk_size, overlap)


    elif clean_model == 'inst_gaboxBV2 (by Gabox)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/inst_gabox.yaml'
          start_check_point = 'ckpts/inst_gaboxBv2.ckpt'
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxBv2.ckpt')
          conf_edit(config_path, chunk_size, overlap)     


    elif clean_model == 'inst_gaboxBFV1 (by Gabox)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/inst_gabox.yaml'
          start_check_point = 'ckpts/gaboxFv1.ckpt'
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxFv1.ckpt')
          conf_edit(config_path, chunk_size, overlap)


    elif clean_model == 'inst_gaboxFV2 (by Gabox)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/inst_gabox.yaml'
          start_check_point = 'ckpts/inst_gaboxFv2.ckpt'
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxFv2.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    
    elif clean_model == 'VOCALS-Male Female-BS-RoFormer Male Female Beta 7_2889 (by aufr33)':
          model_type = 'bs_roformer'
          config_path = 'ckpts/config_chorus_male_female_bs_roformer.yaml'
          start_check_point = 'ckpts/bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt'
          download_file('https://huggingface.co/RareSirMix/AIModelRehosting/resolve/main/bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt')
          download_file('https://huggingface.co/Sucial/Chorus_Male_Female_BS_Roformer/resolve/main/config_chorus_male_female_bs_roformer.yaml')
          conf_edit(config_path, chunk_size, overlap)


    elif clean_model == 'VOCALS-MelBand-Roformer Kim FT 2 (by Unwa)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/config_kimmel_unwa_ft.yaml'
          start_check_point = 'ckpts/kimmel_unwa_ft2.ckpt'
          download_file('https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml')
          download_file('https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft2.ckpt')
          conf_edit(config_path, chunk_size, overlap)      

    elif clean_model == 'voc_gaboxBSroformer (by Gabox)':
          model_type = 'bs_roformer'
          config_path = 'ckpts/voc_gaboxBSroformer.yaml'
          start_check_point = 'ckpts/voc_gaboxBSR.ckpt'
          download_file('https://huggingface.co/GaboxR67/BSRoformerVocTest/resolve/main/voc_gaboxBSroformer.yaml')
          download_file('https://huggingface.co/GaboxR67/BSRoformerVocTest/resolve/main/voc_gaboxBSR.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'voc_gaboxMelReformer (by Gabox)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/voc_gabox.yaml'
          start_check_point = 'ckpts/voc_gabox.ckpt'
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'voc_gaboxMelReformerFV1 (by Gabox)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/voc_gabox.yaml'
          start_check_point = 'ckpts/voc_gaboxFv1.ckpt'
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gaboxFv1.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'voc_gaboxMelReformerFV2 (by Gabox)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/voc_gabox.yaml'
          start_check_point = 'ckpts/voc_gaboxFv2.ckpt'
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gaboxFv2.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'inst_GaboxFv3 (by Gabox)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/inst_gabox.yaml'
          start_check_point = 'ckpts/inst_gaboxFv3.ckpt'
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxFv3.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'Intrumental_Gabox (by Gabox)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/inst_gabox.yaml'
          start_check_point = 'ckpts/intrumental_gabox.ckpt'
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/intrumental_gabox.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'inst_Fv4Noise (by Gabox)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/inst_gabox.yaml'
          start_check_point = 'ckpts/inst_Fv4Noise.ckpt'
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_Fv4Noise.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'inst_V5 (by Gabox)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/inst_gabox.yaml'
          start_check_point = 'ckpts/INSTV5.ckpt'
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV5.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'SYH99999/MelBandRoformerSYHFTB1_Model1 (by Amane)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/config.yaml'
          start_check_point = 'ckpts/model.ckpt'
          download_file('https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/config.yaml')
          download_file('https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/model.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'SYH99999/MelBandRoformerSYHFTB1_Model2 (by Amane)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/config.yaml'
          start_check_point = 'ckpts/model2.ckpt'
          download_file('https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/config.yaml')
          download_file('https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/model2.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'SYH99999/MelBandRoformerSYHFTB1_Model3 (by Amane)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/config.yaml'
          start_check_point = 'ckpts/model3.ckpt'
          download_file('https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/config.yaml')
          download_file('https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/model3.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'VOCALS-MelBand-Roformer Kim FT 2 Blendless (by unwa)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/config_kimmel_unwa_ft.yaml'
          start_check_point = 'ckpts/kimmel_unwa_ft2_bleedless.ckpt'
          download_file('https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml')
          download_file('https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft2_bleedless.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'inst_gaboxFV1 (by Gabox)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/inst_gabox.yaml'
          start_check_point = 'ckpts/inst_gaboxFv1.ckpt'
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxFv1.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'inst_gaboxFV6 (by Gabox)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/inst_gabox.yaml'
          start_check_point = 'ckpts/INSTV6.ckpt'
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV6.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'denoisedebleed (by Gabox)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/model_mel_band_roformer_denoise.yaml'
          start_check_point = 'ckpts/denoisedebleed.ckpt'
          download_file('https://huggingface.co/poiqazwsx/melband-roformer-denoise/resolve/main/model_mel_band_roformer_denoise.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/denoisedebleed.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'INSTV5N (by Gabox)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/inst_gabox.yaml'
          start_check_point = 'ckpts/INSTV5N.ckpt'
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV5N.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'Voc_Fv3 (by Gabox)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/voc_gabox.yaml'
          start_check_point = 'ckpts/voc_Fv3.ckpt'
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_Fv3.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'MelBandRoformer4StemFTLarge (SYH99999)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/config.yaml'
          start_check_point = 'ckpts/MelBandRoformer4StemFTLarge.ckpt'
          download_file('https://huggingface.co/SYH99999/MelBandRoformer4StemFTLarge/resolve/main/config.yaml')
          download_file('https://huggingface.co/SYH99999/MelBandRoformer4StemFTLarge/resolve/main/MelBandRoformer4StemFTLarge.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'dereverb_mel_band_roformer_mono (by anvuew)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/dereverb_mel_band_roformer_anvuew.yaml'
          start_check_point = 'ckpts/dereverb_mel_band_roformer_mono_anvuew_sdr_20.4029.ckpt'
          download_file('https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml')
          download_file('https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_mono_anvuew_sdr_20.4029.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'INSTV6N (by Gabox)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/inst_gabox.yaml'
          start_check_point = 'ckpts/INSTV6N.ckpt'
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV6N.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'KaraokeGabox':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/config_mel_band_roformer_karaoke.yaml'
          start_check_point = 'ckpts/KaraokeGabox.ckpt'
          download_file('https://github.com/deton24/Colab-for-new-MDX_UVR_models/releases/download/v1.0.0/config_mel_band_roformer_karaoke.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/blob/main/melbandroformers/experimental/KaraokeGabox.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'FullnessVocalModel (by Amane)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/config.yaml'
          start_check_point = 'ckpts/FullnessVocalModel.ckpt'
          download_file('https://huggingface.co/Aname-Tommy/MelBandRoformers/blob/main/config.yaml')
          download_file('https://huggingface.co/Aname-Tommy/MelBandRoformers/blob/main/FullnessVocalModel.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'Inst_GaboxV7 (by Gabox)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/inst_gabox.yaml'
          start_check_point = 'ckpts/Inst_GaboxV7.ckpt'
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/Inst_GaboxV7.ckpt')
          conf_edit(config_path, chunk_size, overlap)

     
    
    # Diğer modelleri buraya ekleyebilirsiniz (orijinal kodunuzdaki tüm elif blokları)
    
    return model_type, config_path, start_check_point
