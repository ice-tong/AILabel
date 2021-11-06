static_field = 'ail_token'
static_field_label = 'ail_token_label'
var_label_field = 'var_label'
inst_pos_field = 'stmt_idxs'  # instruction positional embedding
op_pos_field = 'op_idxs'  # opcode/operand positional embedding
byte_fields = [f'value_token_{i}' for i in range(8)]
mem_fields = [f'mem_token_{i}' for i in range(8)]

maskable_fields = [static_field] + byte_fields
aux_fields = [inst_pos_field] + [op_pos_field]
non_byte_fields = [static_field] + [inst_pos_field] + [op_pos_field]
fields = non_byte_fields + byte_fields + mem_fields

byte_len = 8
full_attn = False
min_chunk_len = 20
chunk_mask_relax = 0.9
last_layer = -1
cosine_embedding_loss_margin = 0.2