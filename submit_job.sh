#!/bin/bash
#SBATCH --job-name=medgemma_rag
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=short-simple
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=2:00:00

# --- Notificações por E-mail do SLURM ---
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=agsl@cin.ufpe.br

# Criar pasta de logs se não existir
mkdir -p logs

echo "=========================================="
echo "Iniciando Job $SLURM_JOB_ID em $(date)"
echo "Nó de execução: $SLURMD_NODENAME"
echo "=========================================="

# Inicializar e ativar o ambiente Conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate medgemma_env

# Definir o diretório de cache do Hugging Face
export HF_HOME=~/.cache/huggingface
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || cat ~/.cache/huggingface/stored_tokens 2>/dev/null)
# Executar o pipeline principal
python -u -m src.main

EXIT_CODE=$?

# --- Verificação de Status Final ---
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo " [SUCESSO] Pipeline finalizado com êxito em $(date)!"
else
    echo " [ERRO] O pipeline falhou com código $EXIT_CODE em $(date)!"
    echo "Consulte o arquivo logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err para mais detalhes."
fi

echo "=========================================="

exit $EXIT_CODE
