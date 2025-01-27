from synergyml.config import SynergyMLConfig

# OpenAI Configuration
SynergyMLConfig.set_openai_key("your-key")
SynergyMLConfig.set_openai_org("your-org")

# Azure Configuration
SynergyMLConfig.set_azure_api_base("your-azure-endpoint")
SynergyMLConfig.set_azure_api_version("2023-05-15")

# Google Cloud Configuration
SynergyMLConfig.set_google_project("your-project-id")

# GGUF Configuration
SynergyMLConfig.set_gguf_max_gpu_layers(0) 