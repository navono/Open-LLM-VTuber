docker-init:
	@echo "Initializing Docker environment..."
	@mkdir -p cache models logs avatars backgrounds characters
	@if [ ! -f mcp_servers.json ]; then \
		echo "Creating empty mcp_servers.json..."; \
		echo '{}' > mcp_servers.json; \
	fi
	@if [ ! -f conf.yaml ]; then \
		echo "Creating conf.yaml from template..."; \
		cp config_templates/conf.default.yaml conf.yaml; \
		echo "Please edit conf.yaml to add your API keys!"; \
	fi
	@echo "Docker environment initialized!"

docker-build:
	@echo "Building Open-LLM-VTuber Docker image with proxy..."
	docker build -f dockerfile \
		--build-arg HTTP_PROXY=http://172.18.32.1:18899 \
		--build-arg HTTPS_PROXY=http://172.18.32.1:18899 \
		--build-arg NO_PROXY=localhost,127.0.0.1 \
		-t open-llm-vtuber:latest .

docker-run:
	@echo "Starting Open-LLM-VTuber container..."
	docker compose -f docker-compose.yml up -d

docker-stop:
	@echo "Stopping Open-LLM-VTuber container..."
	docker compose -f docker-compose.yml down

docker-logs:
	@echo "Showing Open-LLM-VTuber container logs..."
	docker compose -f docker-compose.yml logs -f

docker-restart:
	@echo "Restarting Open-LLM-VTuber container..."
	docker compose -f docker-compose.yml restart

docker-clean:
	@echo "Cleaning up Docker resources..."
	docker compose -f docker-compose.yml down -v
	docker rmi open-llm-vtuber:latest || true

.PHONY: docker-init docker-build docker-run docker-stop docker-logs docker-restart docker-clean