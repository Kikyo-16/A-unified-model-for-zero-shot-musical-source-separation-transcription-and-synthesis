from conf.feature import *

QUERY_DIM = 6

MODEL_CONFIG = {"UNet":
		{"with_bn0" : True,
			"input_channels_num" : CHANNELS_NUM,
			"input_size" : WINDOW_SIZE // 2,
			"blocks_num" : 5,
			"condition_dim" : QUERY_DIM,
			"output_dim" : CHANNELS_NUM,
		},
	"QueryNet":
		{"blocks_num" : 2,
			"input_size" : WINDOW_SIZE // 2,
			"input_channels_num" : CHANNELS_NUM,
			"pnum" : QUERY_DIM,
		},
	"Transcriptor":
		{"blocks_num" : 2,
			"output_dim" : NOTES_NUM
		}
}

