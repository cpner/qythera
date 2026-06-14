from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.identity import DefaultAzureCredential

class AzureDeployer:
    def __init__(self, subscription_id, resource_group='qythera-rg'):
        self.credential = DefaultAzureCredential()
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.compute_client = ComputeManagementClient(self.credential, subscription_id)

    def create_gpu_vm(self, vm_name='qythera-gpu', location='eastus'):
        print(f'Creating Azure GPU VM: {vm_name} in {location}')
        poller = self.compute_client.virtual_machines.begin_create_or_update(
            self.resource_group, vm_name,
            {
                'location': location,
                'hardware_profile': {'vm_size': 'Standard_NC6s_v3'},
                'storage_profile': {
                    'image_reference': {'publisher': 'nvidia', 'offer': 'nvidia-cuda',
                                        'sku': '12-0-cuda', 'version': 'latest'},
                    'os_disk': {'create_option': 'FromImage', 'disk_size_gb': 128},
                },
                'network_profile': {'network_interfaces': [{'id': '/subscriptions/...'}]},
            },
        )
        poller.result()
        print(f'VM {vm_name} created')
        return vm_name
