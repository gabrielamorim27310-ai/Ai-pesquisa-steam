#!/bin/bash
# Instala dependências e verifica configuração

echo "📦 Instalando dependências..."
pip install -r requirements.txt

echo ""
if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "⚠️  ANTHROPIC_API_KEY não definida."
  echo "   Execute: export ANTHROPIC_API_KEY='sua-chave-aqui'"
else
  echo "✅ ANTHROPIC_API_KEY encontrada."
fi

echo ""
echo "✅ Pronto! Execute o agente com: python agent.py"
