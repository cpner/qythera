from google.cloud import compute_v1
from google.cloud import storage

class GCPDeployer:
    def __init__(self, project_id):
        self.project_id = project_id
        self.instances_client = compute_v1.InstancesClient()

    def create_gpu_instance(self, zone='us-central1-a', machine_type='a2-highgpu-1g'):
        instance = compute_v1.Instance()
        instance.name = 'qythera-gpu'
        instance.machine_type = f'zones/{zone}/machineTypes/{machine_type}'
        disk = compute_v1.AttachedDisk()
        disk.boot = True
        disk.auto_delete = True
        initialize_params = compute_v1.AttachedDiskInitializeParams()
        initialize_params.source_image = 'projects/deeplearning-platform-release/global/images/common-cu121-v20240101'
        initialize_params.disk_size_gb = 100
        disk.initialize_params = initialize_params
        instance.disks = [disk]
        network_interface = compute_v1.NetworkInterface()
        network_interface.name = 'global/networks/default'
        access_config = compute_v1.AccessConfig()
        network_interface.access_configs = [access_config]
        instance.network_interfaces = [network_interface]
        operation = self.instances_client.insert(
            project=self.project_id, zone=zone, instance_resource=instance
        )
        print(f'Creating instance in {zone}...')
        return operation
