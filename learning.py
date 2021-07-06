import chess.svg
import chess
import numpy as np
import random
import matplotlib.pyplot as plt

from datetime import datetime


# color = True for the White point of view, color = False for the Black point of view
def convert_board_to_int_list(board, color):
    white_list = [0] * 64
    black_list = [0] * 64

    for sq in chess.scan_reversed(board.occupied_co[chess.WHITE]):
        white_list[sq] = board.piece_type_at(sq)

    for sq in chess.scan_reversed(board.occupied_co[chess.BLACK]):
        black_list[sq] = board.piece_type_at(sq)

    if color:  # For the White point of view
        list_board = white_list + black_list
    else:  # For the Black point of view
        list_board = black_list + white_list

    return np.array(list_board)


def compute_position_reward(position_int_list):
    reward = 0

    # Definition of the value of the pieces
    pawn_power = 10
    knight_power = 35
    bishop_power = 35
    rook_power = 50
    queen_power = 90
    king_power = 0

    for i in range(0, 128):
        if position_int_list[i] != 0:
            if i < 64:
                if position_int_list[i] == 1:
                    reward += pawn_power
                elif position_int_list[i] == 2:
                    reward += knight_power
                elif position_int_list[i] == 3:
                    reward += bishop_power
                elif position_int_list[i] == 4:
                    reward += rook_power
                elif position_int_list[i] == 5:
                    reward += queen_power
                elif position_int_list[i] == 6:
                    reward += king_power
            else:
                if position_int_list[i] == 1:
                    reward -= pawn_power
                elif position_int_list[i] == 2:
                    reward -= knight_power
                elif position_int_list[i] == 3:
                    reward -= bishop_power
                elif position_int_list[i] == 4:
                    reward -= rook_power
                elif position_int_list[i] == 5:
                    reward -= queen_power
                elif position_int_list[i] == 6:
                    reward -= king_power

    return float(reward)


def is_end_game(board):
    return board.is_stalemate() or board.is_insufficient_material() or board.can_claim_threefold_repetition()


def play_random_move(board):
    list_legal_move = list(board.legal_moves)

    if len(list_legal_move) > 0:
        random_int = random.randint(0, len(list_legal_move) - 1)
        board.push(list_legal_move[random_int])


def get_list_possibility(board):
    list_legal_move = list(board.legal_moves)
    number_possibility = len(list_legal_move)
    list_position = []

    # Get the color player
    color_player = board.turn

    # For each possibility get the position
    for move in list_legal_move:
        board.push(move)  # We play the move
        list_position.append(convert_board_to_int_list(board, color_player))
        board.pop()  # We cancels the move

    return number_possibility, list_legal_move, list_position


def save_board(board, name, path=""):
    boardsvg = chess.svg.board(board)
    outputfile = open(str(path) + name + ".svg", "w")
    outputfile.write(boardsvg)
    outputfile.close()


def reshape_dataset(dataset, list_number):
    new_dataset = []
    counter_list_number = 0
    counter = 0
    data_prov = []

    for i in range(len(dataset)):
        while counter >= list_number[counter_list_number]:
            data_prov = np.array(data_prov)
            new_dataset.append(data_prov)
            data_prov = []
            counter = 0
            counter_list_number += 1

        data_prov.append(dataset[i])
        counter += 1

    data_prov = np.array(data_prov)
    new_dataset.append(data_prov)

    int_prov = 0
    for i in range(len(new_dataset)):
        int_prov += len(new_dataset[i])
    # print("dataset", len(dataset))
    # print("new_dataset", int_prov)

    return new_dataset


def play_moves_learning(list_chess_board, starting_position, model, coef_exp_int):
    # --- Play move in each parallel game ---
    list_chess_board_play_best_number_possibility = list()
    list_chess_board_play_best_move = list()  # List of game where we play the best move based on the model prediction
    list_best_move_list_move = list()
    list_best_move_position = list()

    for chess_board in list_chess_board:

        # If game is over start new game
        if is_end_game(chess_board):
            list_chess_board[list_chess_board.index(chess_board)] = chess.Board(starting_position)

        # Play move
        random_var = random.uniform(0, 1)

        if random_var <= coef_exp_int:  # Play random move
            play_random_move(chess_board)

        else:  # Prepare to play the best move
            # Get all possibility from the position
            number_possibility, list_legal_move, list_position = get_list_possibility(chess_board)

            list_chess_board_play_best_move.append(chess_board)

            list_chess_board_play_best_number_possibility.append(number_possibility)
            list_best_move_list_move.append(list_legal_move)
            list_best_move_position += list_position

    if len(list_best_move_position) > 0:

        # Predict the value for each possible position for play the best move
        list_best_move_position = np.array(list_best_move_position)
        value_best_move_position = model.predict(list_best_move_position)
        value_best_move_position = reshape_dataset(value_best_move_position,
                                                   list_chess_board_play_best_number_possibility)

        # Play all best move
        for j in range(0, len(value_best_move_position)):
            if len(list_best_move_list_move[j]) > 0:
                list_chess_board_play_best_move[j].push(list_best_move_list_move[j][np.argmax(value_best_move_position[j])])


def create_dataset(list_chess_board, model, coef_expectation, verbose, index_learning_loop):
    list_dataset_X = list()  # List position of game
    list_dataset_Y = list()  # List prediction of game

    # Separated the valid game from the finished game
    list_chess_board_valid = list()
    list_chess_board_end = list()

    for chess_board in list_chess_board:
        if not is_end_game(chess_board):
            list_chess_board_valid.append(chess_board)
        else:
            list_chess_board_end.append(chess_board)

    # --- For the none end game ---
    list_target_position_for_predict = []
    list_target_value_for_predict = []
    list_number_possibility = []

    for chess_board in list_chess_board_valid:
        # Create lists of possibility
        number_possibility, list_legal_move, list_position = get_list_possibility(chess_board)
        list_number_possibility.append(number_possibility)
        list_target_position_for_predict += list_position

        # Compute position reward
        list_position_reward = []
        for j in range(len(list_position)):
            list_position_reward.append(compute_position_reward(list_position[j]))
        list_target_value_for_predict.append(list_position_reward)

    # Add expectation to the value of position
    if coef_expectation > 0 and len(list_target_position_for_predict) > 0:
        list_target_position_for_predict = np.array(list_target_position_for_predict)
        value_predict_position = model.predict(list_target_position_for_predict)
        value_predict_position = reshape_dataset(value_predict_position, list_number_possibility)

        for j in range(len(list_target_value_for_predict)):
            for k in range(len(list_target_value_for_predict[j])):
                list_target_value_for_predict[j][k] += coef_expectation * value_predict_position[j][k]

    # Add the none end game data to the dataset
    for j in range(len(list_chess_board_valid)):
        list_dataset_X.append(convert_board_to_int_list(list_chess_board_valid[j], not list_chess_board_valid[j].turn))
        list_dataset_Y.append(list_target_value_for_predict[j][np.argmin(list_target_value_for_predict[j])])

    # Add the end game data to the dataset
    for j in range(len(list_chess_board_end)):
        list_dataset_X.append(convert_board_to_int_list(list_chess_board_end[j], not list_chess_board_end[j].turn))
        list_dataset_Y.append(np.array([float(0)]))

    if "save_picture_dataset" in verbose:
        counter = 0
        for chess_board in list_chess_board:
            save_board(chess_board, "dataset_game_" + str(counter) + "_loop_" + str(index_learning_loop) + "_val_" + str(0),
                       path="debug_file/")
            counter += 1

    list_dataset_X = np.array(list_dataset_X)
    list_dataset_Y = np.array(list_dataset_Y)

    return list_dataset_X, list_dataset_Y

def learning(starting_position, model, number_cpu_use=1, number_parallel_game=1, number_repetitions=1, window_size=1,
             coef_exp_int=0.3, verbose=[], coef_expectation=0.9):
    if len(verbose) > 0:
        print("--- Learning start ---")

    # start time marker for the learning
    start_time_learning = datetime.now()

    list_chess_board = list()  # List with parallel game

    # We create each parallel game
    for i in range(0, number_parallel_game):
        list_chess_board.append(chess.Board(starting_position))

    # --- Learning loop ---
    for i in range(0, number_repetitions):
        list_dataset_X = list()  # List position of game
        list_dataset_Y = list()  # List prediction of game

        # Play move in each parallel game
        play_moves_learning(list_chess_board=list_chess_board,
                            starting_position=starting_position,
                            model=model,
                            coef_exp_int=coef_exp_int)

        # Create the dataset
        list_dataset_X, list_dataset_Y = create_dataset(
            list_chess_board=list_chess_board,
            model=model,
            coef_expectation=coef_expectation,
            verbose=verbose,
            index_learning_loop=i)

        history = model.fit(list_dataset_X, list_dataset_Y, epochs=5, batch_size=60, shuffle=True, validation_split=0.1, verbose=0)

        if "save_history" in verbose:
            # summarize history for accuracy
            plt.plot(history.history['mean_squared_error'])
            plt.plot(history.history['val_loss'])
            plt.title('model accuracy ' + str(i))
            plt.ylabel('mean squared error')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig('loss_evolution/repetitions_' + str(i) + '.png')
            plt.close()

    # End time marker for the learning
    end_time_learning = datetime.now()
    if "time_learning" in verbose:
        print('Duration: {}'.format(end_time_learning - start_time_learning))

    if len(verbose) > 0:
        print("--- Learning end ---")
        print()