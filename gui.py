import gradio as gr
from download import download_callback
from processing import process_audio, ensemble_audio_fn, auto_ensemble_process
from helpers import clear_old_output, INPUT_DIR, OUTPUT_DIR, generate_random_port, save_uploaded_file

# Model kategorileri ve seçenekleri (örnek, tam listeyi orijinal kodunuzdan alabilirsiniz)
model_choices = {
    "Vocal Separation": ["VOCALS-InstVocHQ", "VOCALS-MelBand-Roformer"],
    "Instrumental Separation": ["INST-Mel-Roformer v1"],
}

def create_interface():
    css = """body { background-color: #2d0b0b; color: #C0C0C0; font-family: 'Poppins', sans-serif; }"""  # Tam CSS kodunuzu buraya ekleyin

    with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
        with gr.Column():
            gr.HTML("<div class='header-text'>Gecekondu Dubbing Production</div>")
        
        with gr.Tabs():
            with gr.Tab("Audio Separation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_audio_file = gr.File(label="Upload Audio", file_types=[".wav", ".mp3"])
                        model_dropdown = gr.Dropdown(label="Model", choices=["VOCALS-InstVocHQ"])  # Tüm modelleri ekleyin
                        chunk_size = gr.Dropdown(label="Chunk Size", choices=[352800, 485100], value=352800)
                        overlap = gr.Slider(2, 50, step=1, label="Overlap", value=2)
                        export_format = gr.Dropdown(label="Format", choices=['wav FLOAT', 'flac PCM_16', 'flac PCM_24'], value='wav FLOAT')
                        use_tta = gr.Checkbox(label="TTA Boost")
                        use_demud_phaseremix_inst = gr.Checkbox(label="Phase Fix")
                        extract_instrumental = gr.Checkbox(label="Instrumental")
                        process_btn = gr.Button("🚀 Process")
                        clear_old_output_btn = gr.Button("🧹 Reset")
                        clear_old_output_status = gr.Textbox(label="Status", interactive=False)
                    with gr.Column(scale=2):
                        original_audio = gr.Audio(label="Original")
                        vocals_audio = gr.Audio(label="Vocals")
                        instrumental_audio = gr.Audio(label="Instrumental")
                        phaseremix_audio = gr.Audio(label="Phase Remix")
                        drum_audio = gr.Audio(label="Drums")
                        karaoke_audio = gr.Audio(label="Karaoke")
                        bass_audio = gr.Audio(label="Bass")
                        other_audio = gr.Audio(label="Other")
                        effects_audio = gr.Audio(label="Effects")
                        speech_audio = gr.Audio(label="Speech")
                        bleed_audio = gr.Audio(label="Bleed")
                        music_audio = gr.Audio(label="Music")
                        dry_audio = gr.Audio(label="Dry")
                        male_audio = gr.Audio(label="Male")
                        female_audio = gr.Audio(label="Female")

            with gr.Tab("Auto Ensemble"):
                with gr.Row():
                    with gr.Column():
                        auto_input_audio_file = gr.File(label="Upload file")
                        auto_file_path_input = gr.Textbox(label="Or enter file path", placeholder="Enter full path to audio file")
                        with gr.Accordion("⚙️ Advanced Settings", open=False):
                            auto_use_tta = gr.Checkbox(label="Use TTA", value=False)
                            auto_extract_instrumental = gr.Checkbox(label="Instrumental Only")
                            auto_overlap = gr.Slider(label="Overlap", minimum=2, maximum=50, value=2, step=1)
                            auto_chunk_size = gr.Dropdown(label="Chunk Size", choices=[352800, 485100], value=352800)
                            export_format2 = gr.Dropdown(label="Output Format", choices=['wav FLOAT', 'flac PCM_16', 'flac PCM_24'], value='wav FLOAT')
                        with gr.Group():
                            gr.Markdown("### 🧠 Model Selection")
                            auto_category_dropdown = gr.Dropdown(label="Model Category", choices=list(model_choices.keys()), value="Vocal Separation")
                            auto_model_dropdown = gr.Dropdown(label="Select Models", choices=model_choices["Vocal Separation"], multiselect=True)
                            selected_models = gr.Dropdown(label="Selected Models", choices=[], multiselect=True, interactive=False)
                            with gr.Row():
                                add_btn = gr.Button("➕ Add Selected")
                                clear_btn = gr.Button("🗑️ Clear All")
                        with gr.Group():
                            gr.Markdown("### ⚡ Ensemble Settings")
                            auto_ensemble_type = gr.Dropdown(label="Method", choices=['avg_wave', 'median_wave', 'min_wave', 'max_wave', 'avg_fft', 'median_fft', 'min_fft', 'max_fft'], value='avg_wave')
                        auto_process_btn = gr.Button("🚀 Start Processing")
                    with gr.Column():
                        original_audio2 = gr.Audio(label="Original Audio")
                        auto_output_audio = gr.Audio(label="Output Preview")
                        auto_status = gr.Textbox(label="Processing Status", placeholder="Waiting for processing...")

            with gr.Tab("Download Sources"):
                with gr.Row():
                    with gr.Column():
                        drive_url_input = gr.Textbox(label="Google Drive URL")
                        drive_download_btn = gr.Button("⬇️ Download from Drive")
                        drive_download_status = gr.Textbox(label="Status")
                        drive_download_output = gr.File(label="Downloaded File")
                    with gr.Column():
                        direct_url_input = gr.Textbox(label="Audio File URL")
                        cookie_file = gr.File(label="Upload Cookies.txt", file_types=[".txt"])
                        direct_download_btn = gr.Button("⬇️ Download from URL")
                        direct_download_status = gr.Textbox(label="Status")
                        direct_download_output = gr.File(label="Downloaded File")

            with gr.Tab("Manuel Ensemble"):
                with gr.Row():
                    with gr.Column(scale=1):
                        refresh_btn = gr.Button("🔄 Refresh")
                        ensemble_type = gr.Dropdown(
                            label="Ensemble Algorithm",
                            choices=['avg_wave', 'median_wave', 'min_wave', 'max_wave', 'avg_fft', 'median_fft', 'min_fft', 'max_fft'],
                            value='avg_wave'
                        )
                        file_dropdown = gr.Dropdown(
                            choices=glob.glob(f"{OUTPUT_DIR}/*.wav") + glob.glob(f"{OLD_OUTPUT_DIR}/*.wav"),
                            label="Available Files",
                            multiselect=True
                        )
                        weights_input = gr.Textbox(label="Custom Weights (comma separated)", placeholder="0.8, 1.2, 1.0")
                    with gr.Column(scale=2):
                        ensemble_output_audio = gr.Audio(label="Ensembled Output")
                        ensemble_status = gr.Textbox(label="Processing Details")
                        ensemble_process_btn = gr.Button("⚡ Process Ensemble")

        # Etkileşimler
        def handle_file_upload(file_obj, file_path_input, is_auto_ensemble=False):
            try:
                existing_files = os.listdir(INPUT_DIR)
                new_file = None
                if file_path_input and os.path.exists(file_path_input):
                    new_file = file_path_input
                elif file_obj:
                    new_file = file_obj.name
                if not new_file and existing_files:
                    kept_file = os.path.join(INPUT_DIR, existing_files[0])
                    return [gr.File(value=kept_file), gr.Audio(value=kept_file)]
                if new_file:
                    clear_directory(INPUT_DIR)
                    saved_path = save_uploaded_file(new_file, is_input=True)
                    return [gr.File(value=saved_path), gr.Audio(value=saved_path)]
                return [None, None]
            except Exception as e:
                print(f"Error: {str(e)}")
                return [None, None]

        def update_models(category):
            return gr.Dropdown(choices=model_choices[category])

        def add_models(new_models, existing_models):
            updated = list(set(existing_models + new_models))
            return gr.Dropdown(choices=updated, value=updated)

        def clear_models():
            return gr.Dropdown(choices=[], value=[])

        process_btn.click(
            fn=process_audio,
            inputs=[input_audio_file, model_dropdown, chunk_size, overlap, export_format, use_tta, use_demud_phaseremix_inst, extract_instrumental, gr.State(None)],
            outputs=[vocals_audio, instrumental_audio, phaseremix_audio, drum_audio, karaoke_audio, bass_audio, other_audio, effects_audio, speech_audio, bleed_audio, music_audio, dry_audio, male_audio, female_audio]
        )
        clear_old_output_btn.click(fn=clear_old_output, outputs=clear_old_output_status)
        drive_download_btn.click(
            fn=download_callback,
            inputs=[drive_url_input, gr.State('drive')],
            outputs=[drive_download_output, drive_download_status, input_audio_file, auto_input_audio_file, original_audio, original_audio2]
        )
        direct_download_btn.click(
            fn=download_callback,
            inputs=[direct_url_input, gr.State('direct'), cookie_file],
            outputs=[direct_download_output, direct_download_status, input_audio_file, auto_input_audio_file, original_audio, original_audio2]
        )
        refresh_btn.click(fn=lambda: gr.Dropdown(choices=glob.glob(f"{OUTPUT_DIR}/*.wav") + glob.glob(f"{OLD_OUTPUT_DIR}/*.wav")), outputs=file_dropdown)
        ensemble_process_btn.click(fn=ensemble_audio_fn, inputs=[file_dropdown, ensemble_type, weights_input], outputs=[ensemble_output_audio, ensemble_status])
        
        input_audio_file.upload(fn=handle_file_upload, inputs=[input_audio_file, gr.State(None)], outputs=[input_audio_file, original_audio])
        auto_input_audio_file.upload(fn=handle_file_upload, inputs=[auto_input_audio_file, auto_file_path_input], outputs=[auto_input_audio_file, original_audio2])
        auto_file_path_input.change(fn=handle_file_upload, inputs=[auto_input_audio_file, auto_file_path_input], outputs=[auto_input_audio_file, original_audio2])
        auto_category_dropdown.change(fn=update_models, inputs=auto_category_dropdown, outputs=auto_model_dropdown)
        add_btn.click(fn=add_models, inputs=[auto_model_dropdown, selected_models], outputs=selected_models)
        clear_btn.click(fn=clear_models, inputs=[], outputs=selected_models)
        auto_process_btn.click(
            fn=auto_ensemble_process,
            inputs=[auto_input_audio_file, selected_models, auto_chunk_size, auto_overlap, export_format2, auto_use_tta, auto_extract_instrumental, auto_ensemble_type, gr.State(None)],
            outputs=[auto_output_audio, auto_status]
        )

    return demo
