import logging

import json
from ruamel.yaml import YAML

logger = logging.getLogger(__name__)

personality_index = {}
personality_dir = 'config/personality/'

with open(f'{personality_dir}index.yaml') as stream:
	yaml = YAML(typ='safe', pure=True)
	personality_index = yaml.load(stream)

class Reminder:
	def __init__(self):
		self.unix_time
		self.description

#TODO: use tiktoken to ensure context length.
class Instance:
	def __init__(self, unique_id):
		self.unique_id 			= unique_id
		self.always_on			= True
		# Personality
		self.personality		= 'lockedin'
		self.modalities			= ['text', 'image']
		self.model 				= None
		self.system_prompt 		= None
		self.custom_instruction = None
		self.enable_vision		= False
		self.context_length		= 16
		self.temperature		= 1.0

		self.memory 			= []
		self.context_window		= []

		self.zone_offset		= 8

		self._load_config()
		self.set_personality(self.personality)

	def set_personality(self, key):
		self.personality = key
		file = personality_dir + personality_index[key]
		with open(file) as stream:
			yaml = YAML(typ='safe', pure=True)
			root = yaml.load(stream)
			self.model = root['model']
			self.temperature = root['temp']
			self.context_length = root['context_length']
			self.custom_instruction = root['custom_instruction']

	def get_context(self):
		return self._get_system_prompt()+self.context_window

	# Adding to context; convenience functions
	def add_to_context(self, message):
		self.context_window.append(message)
		self.context_window = self.context_window[-self.context_length:]
		last_chat = self.context_window[0]
		if last_chat['role'] == 'tool':
			# Remove both tool_calls and tool 
			self.context_window = self.context_window[1:]

	def add_user_message(self, text, image_url=None):
		message = {
			'role': 'user',
			'content': []
		}

		if text:
			message['content'].append(
				{
					'type': 'text',
					'text': text
				}
			)
		if image_url and False:
			message['content'].append(
				{
					'type': 'image_url',
					'image_url': {
						'url': image_url,
						"detail": "low"
					}
				}
			)

		self.add_to_context(message)
		logger.info(f'{self.context_window[-1]['content'][0]['text']}\n')

	def add_assistant_message(self, text):
		message = {
			'role': 'assistant',
			'content': text,
		}

		self.add_to_context(message)
		logger.info(f'RotBot: {self.context_window[-1]['content']}\n')

	def add_system_message(self, text):
		message = {
			'role': 'system',
			'content': text,
		}

		self.add_to_context(message)
		logger.info(f'RotBot: {self.context_window[-1]['content']}\n')

	def add_tool_call(self, tool_call_id, tool_name, tool_arguments):
		tool_call_message = {
			'role': 'assistant',
			'tool_calls': [
				{
				'id': tool_call_id,
				'type': 'function',
				'function': {
						'name': tool_name,
						'arguments': json.dumps(tool_arguments),
					}
				}
			]
		}

		self.add_to_context(tool_call_message)

	def add_tool_response(self, tool_call_id, tool_response):
		tool_call_response = {
			'role' : 'tool',
			'tool_call_id': tool_call_id,
			'content' : json.dumps(tool_response),
		}

		self.add_to_context(tool_call_response)

	# def add_to_memory(content):
	# 	self.memory.append(content)
	# 	with open(f'data/memory/{self.unique_id}.json', 'w') as file:
	# 		file.write(json.dumps(self.memory, indent=4))

	# Private
	
	def _load_config(self):
		with open('config/base.yaml') as stream:
			yaml = YAML(typ='safe', pure=True)
			root = yaml.load(stream)
			self.model = root['model']
			self.temperature = root['temp']
			self.system_prompt = root['system_prompt']
			self.context_length = root['context_length']

	def _get_system_prompt(self):
		processed_system_prompt = self.system_prompt
		# processed_system_prompt = processed_system_prompt.replace('MEMORY', self.get_memory(chat_id))
		return [
			{
				'role': 'system', 
				'content': processed_system_prompt+self.custom_instruction
			},
		]
