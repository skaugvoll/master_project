import yaml


path = '../params/test_config.yml'
config_file = open(path, 'r')
data = yaml.load(config_file)
print(data)
print(type(data))

print()
for k, v in data.items():
    print(k, v)