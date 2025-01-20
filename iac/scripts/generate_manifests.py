#!/usr/bin/env python3
import yaml
import os
from pathlib import Path

def load_config():
    """Load repository configuration with environment variable substitution."""
    config_path = Path(__file__).parent.parent / "config" / "repository.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Substitute environment variables
    registry = os.getenv("CONTAINER_REGISTRY", config["container_registry"]["host"])
    namespace = os.getenv("REGISTRY_NAMESPACE", config["container_registry"]["namespace"])
    tag = os.getenv("IMAGE_TAG", config["container_registry"]["tag"])
    
    return {
        "registry": registry,
        "namespace": namespace,
        "tag": tag,
        "services": config["services"]
    }

def generate_image_name(registry, namespace, service, tag):
    """Generate full image name."""
    return f"{registry}/{namespace}/{service}:{tag}"

def generate_manifests(config):
    """Generate Kubernetes manifests for all services."""
    k8s_dir = Path(__file__).parent.parent / "k8s"
    k8s_dir.mkdir(exist_ok=True)
    
    # Generate manifests for each service
    for service in config["services"]:
        name = service["name"]
        image = generate_image_name(
            config["registry"],
            config["namespace"],
            name,
            config["tag"]
        )
        
        if name == "training":
            manifest = generate_training_manifest(name, image)
        else:
            manifest = generate_service_manifest(
                name,
                image,
                service["port"],
                service["target_port"]
            )
        
        # Write manifest to file
        manifest_path = k8s_dir / f"{name}.yaml"
        with open(manifest_path, "w") as f:
            yaml.dump(manifest, f)

def generate_service_manifest(name, image, port, target_port):
    """Generate manifest for regular services."""
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": name},
        "spec": {
            "replicas": 2 if name == "serving" else 1,
            "selector": {"matchLabels": {"app": name}},
            "template": {
                "metadata": {"labels": {"app": name}},
                "spec": {
                    "containers": [{
                        "name": name,
                        "image": image,
                        "resources": {
                            "limits": {"nvidia.com/gpu": 1}
                        } if name == "serving" else {},
                        "ports": [{"containerPort": target_port}],
                        "env": [
                            {"name": "DB_PATH", "value": "/data/scenarios.db"}
                        ] if name == "data-curation" else [
                            {"name": "MODEL_DIR", "value": "/data/whisper_finetuned"}
                        ] if name == "serving" else [],
                        "volumeMounts": [{
                            "name": "data-volume",
                            "mountPath": "/data"
                        }]
                    }],
                    "volumes": [{
                        "name": "data-volume",
                        "persistentVolumeClaim": {"claimName": "data-pvc"}
                    }]
                }
            }
        }
    }

def generate_training_manifest(name, image):
    """Generate manifest for training job."""
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": "whisper-training"},
        "spec": {
            "template": {
                "spec": {
                    "containers": [{
                        "name": name,
                        "image": image,
                        "resources": {
                            "limits": {"nvidia.com/gpu": 1}
                        },
                        "env": [
                            {"name": "DATASET_PATH", "value": "/data/training_data/training_data.json"},
                            {"name": "MODEL_NAME", "value": "openai/whisper-small"},
                            {"name": "OUTPUT_DIR", "value": "/data/whisper_finetuned"}
                        ],
                        "volumeMounts": [{
                            "name": "data-volume",
                            "mountPath": "/data"
                        }]
                    }],
                    "volumes": [{
                        "name": "data-volume",
                        "persistentVolumeClaim": {"claimName": "data-pvc"}
                    }],
                    "restartPolicy": "Never"
                }
            },
            "backoffLimit": 4
        }
    }

if __name__ == "__main__":
    config = load_config()
    generate_manifests(config)
    print("Kubernetes manifests generated successfully!") 