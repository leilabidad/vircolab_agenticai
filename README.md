# vircolab_agenticai

## Structure

```
 vircolab_agenticai/
│
├── cpu_base/
│   ├── config/
│   │   └── config.yaml
│   │
│   ├── data/
│   │   ├── fake_notes/
│   │   └── mimic_samples/
│   │
│   ├── results/
│   │
│   ├── src/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── image_selector.py
│   │   ├── note_generator.py
│   │   ├── pipeline.py
│   │   └── utils.py
│   │
│   ├── run.py
│   ├── requirements.txt
│   └── README.md
│
├── gpu_base/
│   ├── config/
│   │   └── config.yaml
│   │
│   ├── data/
│   │   ├── clinical_notes/
│   │   └── mimic_samples/
│   │
│   ├── results/
│   │
│   ├── src/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── visit_index.py
│   │   ├── image_selector.py
│   │   ├── note_model.py
│   │   ├── dataset_builder.py
│   │   ├── checkpoint.py
│   │   └── utils.py
│   │
│   ├── run.py
│   ├── requirements.txt
│   └── README.md
│
├── .gitignore
├── pyproject.toml
└── README.md


```
