# 🚀 LitLens v2.0 Deployment Checklist

## Pre-Deployment Preparation

### ✅ Code Ready
- [ ] All modules refactored and tested
- [ ] Frontend integrated and working locally
- [ ] Environment variables configured
- [ ] Dependencies updated in requirements.txt and package.json
- [ ] Documentation complete

### ✅ Repository Setup
- [ ] Git repository initialized
- [ ] .gitignore configured
- [ ] Initial commit created
- [ ] GitHub repository created (recommended: `litlens-v2`)
- [ ] Remote origin added and pushed

### ✅ Configuration Files
- [ ] `vercel.json` configured for monorepo deployment
- [ ] `next.config.mjs` updated with API routing
- [ ] `.env.example` files created
- [ ] `deploy.sh` script ready

## Deployment Options

### Option A: Quick Deployment (Recommended)
```bash
# Use the automated deployment script
./deploy.sh preview    # For testing
./deploy.sh production # For live deployment
```

### Option B: Manual Deployment
```bash
# 1. Install Vercel CLI
npm install -g vercel

# 2. Deploy
vercel        # Preview deployment
vercel --prod # Production deployment
```

### Option C: GitHub Integration (Best for Long-term)
1. Push to GitHub first
2. Connect repository in Vercel dashboard
3. Enable automatic deployments
4. Set up branch-specific deployments

## Environment Variables Setup

### Required Variables for Vercel
```bash
# Set these in Vercel dashboard or CLI
vercel env add OPENAI_API_KEY
vercel env add HUGGINGFACE_API_TOKEN
vercel env add NEXT_PUBLIC_API_URL
vercel env add ENVIRONMENT
```

### Local Development
```bash
# Backend (.env)
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_API_TOKEN=your_huggingface_api_token_here
ENVIRONMENT=development

# Frontend (.env.local)
NEXT_PUBLIC_API_URL=http://localhost:7860
```

### Production Values
```bash
OPENAI_API_KEY=sk-...                          # Your OpenAI API key
HUGGINGFACE_API_TOKEN=hf_...                   # Your HuggingFace token
NEXT_PUBLIC_API_URL=https://your-api-domain    # Backend URL
ENVIRONMENT=production
```

## Deployment Process

### Step 1: Pre-flight Checks
- [ ] Local development working (frontend + backend)
- [ ] All tests passing
- [ ] No build errors
- [ ] Environment variables set
- [ ] Git repository up to date

### Step 2: Deploy to Preview
```bash
./deploy.sh preview
# OR
vercel
```

### Step 3: Test Preview Deployment
- [ ] Frontend loads correctly
- [ ] API endpoints responding
- [ ] File upload working
- [ ] Summarization functioning
- [ ] Error handling working
- [ ] No console errors

### Step 4: Deploy to Production
```bash
./deploy.sh production
# OR
vercel --prod
```

### Step 5: Post-Deployment Verification
- [ ] Production URL accessible
- [ ] All features working
- [ ] Performance acceptable
- [ ] Error monitoring active
- [ ] Custom domain configured (if needed)

## Rollback Strategy

### Instant Rollback (Vercel Dashboard)
1. Go to Vercel dashboard
2. Navigate to Deployments
3. Find previous working deployment
4. Click "Promote to Production"

### Git-based Rollback
```bash
# Revert to previous commit
git revert HEAD
git push origin main

# OR reset to specific commit
git reset --hard [commit-hash]
git push --force origin main
```

### Branch-based Rollback
```bash
# Create rollback branch from working commit
git checkout -b rollback-v1 [working-commit-hash]
git push origin rollback-v1

# Deploy rollback branch in Vercel
vercel --prod
```

## Monitoring & Health Checks

### Health Check Endpoints
- `GET /` - Basic backend health
- `GET /api/health` - Detailed health status
- `GET /docs` - API documentation

### Monitoring Setup
- [ ] Vercel Analytics enabled
- [ ] Error tracking configured
- [ ] Performance monitoring active
- [ ] Uptime monitoring set up
- [ ] Alert notifications configured

## Troubleshooting Common Issues

### Build Failures
```bash
# Clear caches and rebuild
rm -rf node_modules .next
npm install
npm run build
```

### API Connection Issues
- [ ] Check NEXT_PUBLIC_API_URL is set correctly
- [ ] Verify CORS settings
- [ ] Check API endpoint URLs
- [ ] Validate environment variables

### Timeout Issues
- [ ] Increase function timeout in vercel.json
- [ ] Optimize heavy operations
- [ ] Implement proper caching
- [ ] Add request queuing

### Memory Issues
- [ ] Optimize large file processing
- [ ] Implement streaming for PDFs
- [ ] Add memory monitoring
- [ ] Consider upgrading Vercel plan

## Security Checklist

### Environment Variables
- [ ] API keys stored securely in Vercel
- [ ] No sensitive data in client-side code
- [ ] Environment variables properly scoped

### File Upload Security
- [ ] File type validation active
- [ ] File size limits enforced
- [ ] Path traversal protection enabled
- [ ] Temporary file cleanup working

### API Security
- [ ] Input validation implemented
- [ ] Rate limiting configured
- [ ] Error messages sanitized
- [ ] HTTPS enforced

## Performance Optimization

### Frontend
- [ ] Next.js build optimization enabled
- [ ] Image optimization configured
- [ ] Bundle size acceptable
- [ ] Lazy loading implemented

### Backend
- [ ] Response compression enabled
- [ ] Caching strategies implemented
- [ ] Database queries optimized
- [ ] Memory usage monitored

## Success Criteria

### Functional Requirements
- [ ] PDF upload and processing works
- [ ] AI summarization generates quality output
- [ ] Report generation and download functional
- [ ] Error handling graceful and informative
- [ ] Performance meets requirements

### Technical Requirements
- [ ] Zero downtime deployment capability
- [ ] Rollback procedures tested
- [ ] Monitoring and alerting active
- [ ] Security measures implemented
- [ ] Documentation complete

## Post-Deployment Tasks

### Immediate (Day 1)
- [ ] Verify all functionality
- [ ] Monitor error rates
- [ ] Check performance metrics
- [ ] Test rollback procedures
- [ ] Update documentation

### Short-term (Week 1)
- [ ] Analyze usage patterns
- [ ] Optimize based on real traffic
- [ ] Set up automated monitoring
- [ ] Plan iterative improvements
- [ ] Gather user feedback

### Long-term (Month 1)
- [ ] Performance optimization
- [ ] Feature enhancements
- [ ] Security audit
- [ ] Cost optimization
- [ ] Scale planning

## Emergency Contacts & Resources

### Quick Commands
```bash
# Emergency rollback
vercel --prod --force

# View logs
vercel logs [deployment-url]

# Check deployment status
vercel ls

# Remove deployment
vercel rm [deployment-url]
```

### Support Resources
- Vercel Documentation: https://vercel.com/docs
- Next.js Documentation: https://nextjs.org/docs
- FastAPI Documentation: https://fastapi.tiangolo.com
- LitLens Documentation: ./README.md

---

**Remember**: Always test in preview before promoting to production! 🚀