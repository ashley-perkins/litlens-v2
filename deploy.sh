#!/bin/bash

# LitLens v2.0 Deployment Script
# Usage: ./deploy.sh [preview|production]

set -e

DEPLOY_TYPE=${1:-preview}
REPO_NAME="litlens-v2"

echo "🚀 Starting LitLens v2.0 deployment..."
echo "📋 Deployment type: $DEPLOY_TYPE"

# Check if we're in the right directory
if [ ! -f "vercel.json" ]; then
    echo "❌ Error: vercel.json not found. Are you in the project root?"
    exit 1
fi

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "🔧 Initializing Git repository..."
    git init
    
    # Create .gitignore
    cat > .gitignore << EOF
# Dependencies
node_modules/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Environment variables
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

# Next.js
.next/
out/
build/

# Vercel
.vercel

# Logs
logs
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Runtime data
pids
*.pid
*.seed
*.pid.lock

# Coverage directory used by tools like istanbul
coverage/

# nyc test coverage
.nyc_output

# Dependency directories
jspm_packages/

# Optional npm cache directory
.npm

# Optional REPL history
.node_repl_history

# Output of 'npm pack'
*.tgz

# Yarn Integrity file
.yarn-integrity

# dotenv environment variables file
.env

# Temporary files
tmp/
temp/
*.tmp

# Python specific
*.egg-info/
dist/
build/
.pytest_cache/
.coverage
htmlcov/
.tox/
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/

# Jupyter Notebook
.ipynb_checkpoints

# pyenv
.python-version

# celery beat schedule file
celerybeat-schedule

# SageMath parsed files
*.sage.py

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json
EOF

    echo "✅ Git repository initialized"
fi

# Check for required files
echo "🔍 Checking deployment requirements..."

required_files=("vercel.json" "frontend/package.json" "backend/app.py" "requirements.txt")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Error: Required file $file not found"
        exit 1
    fi
done

echo "✅ All required files present"

# Install dependencies
echo "📦 Installing dependencies..."

# Backend dependencies
if [ ! -d "venv" ]; then
    echo "🐍 Creating Python virtual environment..."
    python -m venv venv
fi

echo "🐍 Installing Python dependencies..."
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate
pip install -r requirements.txt

# Frontend dependencies
echo "📱 Installing Node.js dependencies..."
cd frontend
npm install
cd ..

echo "✅ Dependencies installed"

# Run basic tests
echo "🧪 Running basic validation tests..."

# Test backend import
python -c "from backend.app import app; print('✅ Backend imports successfully')" || {
    echo "❌ Backend import failed"
    exit 1
}

# Test frontend build
cd frontend
npm run build > /dev/null 2>&1 && echo "✅ Frontend builds successfully" || {
    echo "❌ Frontend build failed"
    exit 1
}
cd ..

echo "✅ Basic tests passed"

# Git operations
echo "📝 Preparing Git commit..."

git add .
git commit -m "feat: LitLens v2.0 - Consolidated and refactored

- Merged litlens + litlens-portal repositories
- Refactored all backend modules with enterprise features  
- Enhanced Next.js frontend with improved UX
- Added comprehensive error handling and validation
- Implemented caching, monitoring, and security features
- Maintained 100% backward compatibility
- Ready for production deployment

🚀 Deploy type: $DEPLOY_TYPE" || echo "ℹ️  No changes to commit"

# Check if we have a remote
if ! git remote get-url origin > /dev/null 2>&1; then
    echo "⚠️  No Git remote configured."
    echo "📋 To set up GitHub repository:"
    echo "   1. Create repository at: https://github.com/ashley-perkins/$REPO_NAME"
    echo "   2. Run: git remote add origin https://github.com/ashley-perkins/$REPO_NAME.git"
    echo "   3. Run: git push -u origin main"
    echo ""
    read -p "🤔 Continue with local deployment? (y/N): " continue_local
    if [[ ! $continue_local =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

# Deploy with Vercel
echo "🚀 Deploying to Vercel..."

if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Installing..."
    npm install -g vercel
fi

if [ "$DEPLOY_TYPE" = "production" ]; then
    echo "🌟 Deploying to PRODUCTION..."
    vercel --prod
else
    echo "🔍 Deploying to PREVIEW..."
    vercel
fi

echo ""
echo "🎉 Deployment complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Test the deployment URL"
echo "   2. Set up environment variables in Vercel dashboard"
echo "   3. Configure custom domain if needed"
echo "   4. Set up monitoring and alerts"
echo ""
echo "🔧 Environment variables to set in Vercel:"
echo "   - OPENAI_API_KEY (sensitive)"
echo "   - HUGGINGFACE_API_TOKEN (sensitive)"
echo "   - NEXT_PUBLIC_API_URL (your backend URL)"
echo "   - ENVIRONMENT=production"
echo ""

deactivate 2>/dev/null || true
echo "✅ Done!"