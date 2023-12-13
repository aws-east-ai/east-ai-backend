echo "Deleting ../east-ai-deploy/dist/east-ai-backend"
rm -rf ../east-ai-deploy/dist/east-ai-backend
git clone https://github.com/aws-east-ai/east-ai-backend ../east-ai-deploy/dist/east-ai-backend
rm -rf ../east-ai-deploy/dist/east-ai-backend/.git

cd ../east-ai-deploy/dist/east-ai-backend
git add .
git commit -m "update"
git push