import shared_vars as sv
import helpers.services as serv
import helpers.tel as tel

async def start_of_program_preparing():
    serv.remove_files('_saldos')
    serv.remove_files('_profits')
    serv.delete_folder_contents('_pic')
    sv.global_pos_1m = []
    if sv.settings.prep_data != 'B':
        serv.remove_files('_classif_train_data_1')
    # if sv.settings.prep_data != 'B':
    #         # serv.delete_folder_contents(sv.path_to_save_examples + '0')
    #         # serv.delete_folder_contents(sv.path_to_save_examples + '1')
    #         # serv.delete_folder_contents(sv.path_to_save_examples + '2')
    # else:
    #     msg = (
    #         f"Start with strategy: {sv.settings.prep_data} "
    #     )
    #     await tel.send_inform_message(msg, '', False)