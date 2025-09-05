# DS-peso-y-volumetria-de-mercancias



# Para hacer predicciones:
docker-compose run --rm product-predictor python cli/main.py predict \
  --input /app/data/raw/new_products.csv \
  --output /app/data/processed/predictions.csv

# Modo interactivo:
docker-compose run --rm product-predictor bash
