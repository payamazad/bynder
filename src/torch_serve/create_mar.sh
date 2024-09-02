torch-model-archiver --model-name metadata_creator \
                     --version 1.0 \
                     --serialized-file ../../models/gender/traced_model.pt \
                     --handler handler.py \
                     --export-path model_store/

torch-model-archiver --model-name articleType \
                     --version 1.0 \
                     --serialized-file ../../models/articleType/traced_model.pt \
                     --handler handler.py \
                     --export-path model_store/

torch-model-archiver --model-name baseColour \
                     --version 1.0 \
                     --serialized-file ../../models/baseColour/traced_model.pt \
                     --handler handler.py \
                     --export-path model_store/

torch-model-archiver --model-name season \
                     --version 1.0 \
                     --serialized-file ../../models/season/traced_model.pt \
                     --handler handler.py \
                     --export-path model_store/

torch-model-archiver --model-name usage \
                     --version 1.0 \
                     --serialized-file ../../models/usage/traced_model.pt \
                     --handler handler.py \
                     --export-path model_store

