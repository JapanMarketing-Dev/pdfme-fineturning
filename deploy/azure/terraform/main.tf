# Terraform設定 - Azure VM + GPU
# Qwen3-VL + LoRA APIサーバー

terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features {}
}

variable "resource_group_name" {
  default = "pdfme-form-detector-rg"
}

variable "location" {
  default = "japaneast"
}

variable "vm_size" {
  default = "Standard_NC4as_T4_v3"  # 4 vCPU, 28GB RAM, T4 16GB VRAM
  # 他の選択肢: Standard_NC6s_v3 (V100 16GB), Standard_NC24ads_A100_v4 (A100 80GB)
}

variable "admin_username" {
  default = "azureuser"
}

variable "admin_password" {
  description = "VM管理者パスワード"
  sensitive   = true
}

# リソースグループ
resource "azurerm_resource_group" "pdfme" {
  name     = var.resource_group_name
  location = var.location
}

# 仮想ネットワーク
resource "azurerm_virtual_network" "pdfme" {
  name                = "pdfme-vnet"
  address_space       = ["10.0.0.0/16"]
  location            = azurerm_resource_group.pdfme.location
  resource_group_name = azurerm_resource_group.pdfme.name
}

resource "azurerm_subnet" "pdfme" {
  name                 = "pdfme-subnet"
  resource_group_name  = azurerm_resource_group.pdfme.name
  virtual_network_name = azurerm_virtual_network.pdfme.name
  address_prefixes     = ["10.0.1.0/24"]
}

# パブリックIP
resource "azurerm_public_ip" "pdfme" {
  name                = "pdfme-api-ip"
  location            = azurerm_resource_group.pdfme.location
  resource_group_name = azurerm_resource_group.pdfme.name
  allocation_method   = "Static"
  sku                 = "Standard"
}

# ネットワークセキュリティグループ
resource "azurerm_network_security_group" "pdfme" {
  name                = "pdfme-nsg"
  location            = azurerm_resource_group.pdfme.location
  resource_group_name = azurerm_resource_group.pdfme.name

  security_rule {
    name                       = "SSH"
    priority                   = 1001
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "22"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  security_rule {
    name                       = "API"
    priority                   = 1002
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "8000"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }
}

# ネットワークインターフェース
resource "azurerm_network_interface" "pdfme" {
  name                = "pdfme-nic"
  location            = azurerm_resource_group.pdfme.location
  resource_group_name = azurerm_resource_group.pdfme.name

  ip_configuration {
    name                          = "internal"
    subnet_id                     = azurerm_subnet.pdfme.id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.pdfme.id
  }
}

resource "azurerm_network_interface_security_group_association" "pdfme" {
  network_interface_id      = azurerm_network_interface.pdfme.id
  network_security_group_id = azurerm_network_security_group.pdfme.id
}

# GPU VM
resource "azurerm_linux_virtual_machine" "pdfme" {
  name                = "pdfme-form-detector-vm"
  resource_group_name = azurerm_resource_group.pdfme.name
  location            = azurerm_resource_group.pdfme.location
  size                = var.vm_size
  admin_username      = var.admin_username
  admin_password      = var.admin_password
  disable_password_authentication = false

  network_interface_ids = [azurerm_network_interface.pdfme.id]

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Premium_LRS"
    disk_size_gb         = 100
  }

  source_image_reference {
    publisher = "microsoft-dsvm"
    offer     = "ubuntu-hpc"
    sku       = "2204"
    version   = "latest"
  }

  custom_data = base64encode(<<-EOF
    #!/bin/bash
    set -e
    
    # NVIDIAドライバーインストール
    apt-get update
    apt-get install -y nvidia-driver-535 nvidia-container-toolkit
    
    # Dockerインストール
    apt-get install -y docker.io
    systemctl start docker
    systemctl enable docker
    
    # NVIDIA Container Toolkit設定
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
    
    # APIサーバー起動
    cd /home/${var.admin_username}
    git clone https://github.com/JapanMarketing-Dev/pdfme-fineturning.git || true
    cd pdfme-fineturning/deploy
    docker build -t pdfme-api:latest -f Dockerfile .
    
    docker run -d \
      --name pdfme-api \
      --gpus all \
      -p 8000:8000 \
      -e BASE_MODEL=Qwen/Qwen3-VL-8B-Instruct \
      -e LORA_ADAPTER=takumi123xxx/pdfme-form-field-detector-lora \
      -e USE_4BIT=true \
      --restart unless-stopped \
      pdfme-api:latest
  EOF
  )

  tags = {
    app = "pdfme-form-detector"
  }
}

output "api_url" {
  value = "http://${azurerm_public_ip.pdfme.ip_address}:8000"
}

output "ssh_command" {
  value = "ssh ${var.admin_username}@${azurerm_public_ip.pdfme.ip_address}"
}

output "health_check" {
  value = "curl http://${azurerm_public_ip.pdfme.ip_address}:8000/health"
}

