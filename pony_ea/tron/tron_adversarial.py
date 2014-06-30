#! /usr/bin/env python

# The MIT License (MIT)

# Copyright (c) 2014 Erik Hemberg

# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
One player version of STU Tron.
Author: Erik Hemberg
"""

from Tkinter import TclError
import argparse
import datetime

from ai_tron_non_adversarial import Player, PlayerAI, Tron, \
    read_strategy_from_file


class TronAdversarial(Tron):
    """
    Tron Adversarial represents the game, it has:

     - a board, which is a square toroidal grid
     - a canvas which is used when drawing and displaying the player
     - a player which controls the bike that is constantly moving forward and
       leaves a trail

    **STU Tron** game flow:

    *Initialization*
      The board, player and canvas are initialized. The listeners of the
      canvas are initialized

    *Run*
      Update the player each step of the game. The direction of the player is
      updated by key events. The key events trigger an :ref:`actions` of a
      player:

       - A human player presses the '<' or '>' arrow on the keyboard. The
         direction of the player is changed and the canvas is redrawn.
       - An AI player executes a strategy

    *Cleanup*
      The game ends when the player collides with and obstacle. Close the GUI
      and write the statistics of the game to a log file.

    Attributes:

    - Canvas -- Used for displaying the GUI
    - Rows -- The number of cells on the board
    - Board -- The board where player moves. Keeps track of the trail that
      the bike leaves behind
    - Bike Width -- The width in pixels of the bike. Used when drawing the
      bike on the GUI
    - Draw board -- Indicator for turning the GUI on or off.
    - Players -- The players controlling the bike. The initial position of the
      players are close to the center of the board and the initial direction is
      facing south
    - Strategy -- The predefined strategy of an ai player, if the strategy is
      `None` a human can control the player
    - Winner -- The player who won

    .. note:: Executing **STU Tron** without a GUI is faster. There is no
       need to draw on the canvas and DELAY each step.

    STATS_FILE
      Name of file for saving the statistics from each game

    """

    STATS_FILE = "tron_game_stats.log"

    def __init__(self, rows, bike_width, draw_board, strategy):
        """Constructor"""
        super(TronAdversarial, self).__init__(rows, bike_width, draw_board,
                                              strategy)
        self.winner = None
        # Create human or AI player
        self.players = [
            Player(x=rows / 2, y=rows / 2, direction=(0, 1),
                   color="Blue", alive=True, id_=0,
                   canvas=self.canvas, board=self.board),
            PlayerAI(x=rows / 2 + 3, y=rows / 2, direction=(0, 1),
                     color="Green", alive=True, id_=1,
                     canvas=self.canvas, board=self.board,
                     strategy=strategy)
        ]

    def step(self):
        """
        Function called if GUI not initialized. Performs a step of the
        game:

         - Updates the player
         - Check if the game is over.
        """
        # Make the move of the AI player
        for player in self.players:
            if isinstance(player, PlayerAI):
                self.ai_key_pressed(player)

        if self.game_over is False:
            for player in self.players:
                # Update the player
                player.update()

        for player in self.players:
            # Check if player is alive
            if not player.alive:
                self.game_over = True
                self.write_stats()

    def key_pressed(self, event):
        """
        Determine the action when a key is pressed.

        Key events:

        - Left arrow *<* turns player 90 degrees counter clockwise
        - Right arrow *>* key turns the player 90 degrees clockwise
        - *q* sets the game to be over

        :param event: event on GUI
        :type event: TkInter Event
        """
        # Select the player that is affected by  the key press
        player = self.players[0]

        # Process keys that work even if the game is over
        if event.char == "q":
            self.game_over = True

        # Process keys that only work if the game is not over
        if not self.game_over:
            if event.keysym == "Left":
                player.left()
            elif event.keysym == "Right":
                player.right()

            # Redraw the board
            if self.draw_board:
                self.redraw_all()

    def get_winner(self):
        """
        Set the winner of the STU Tron A game. The player with the longest trail
        wins.
        """
        # Check the trail lengths of the bikes
        if len(self.players[0].bike_trail) > len(self.players[1].bike_trail):
            self.winner = (self.players[0].ID, len(self.players[0].bike_trail))
        elif len(self.players[0].bike_trail) < len(self.players[1].bike_trail):
            self.winner = (self.players[1].ID, len(self.players[1].bike_trail))
        else:
            self.winner = (None, None)

    def write_stats(self):
        """
        Write the statistics of the game to a file when the game is over.

        The statistics recorded are:

        - Time of game
        - Number of rows on the board        
        - ID of player
        - Length of player trail
        - Strategy used by player. (A human player strategy is *None*)
        - ID of player
        - Length of player trail
        - Strategy used by player. (A human player strategy is *None*)
        - Winner

        """
        if self.game_over is True:
            self.get_winner()
            # Write stats to file
            f_out = open(Tron.STATS_FILE, 'a')
            for player in self.players:
                # Set the size of the bike trail
                text_str = 'id %d; board rows %d; bike trail length %s' % \
                           (player.ID, len(self.board), len(player.bike_trail))
                # Get the timestamp
                time_stamp = datetime.datetime.now()
                strategy = None
                if isinstance(player, PlayerAI):
                    strategy = player.strategy

                # Print the information
                print('Time; %s' % time_stamp)
                print('STU Tron Non-Adversarial; %s' % text_str)
                print('GUI; %s' % self.draw_board)
                print('Trail; %s' % player.bike_trail)
                print('Strategy; %s' % strategy)
                # Write, time, rows, id, bike trail length, strategy
                f_out.write('%s, %d, %d, %d, strategy:\n%s\n' %
                            (time_stamp, len(self.board), self.player.ID,
                             len(player.bike_trail), str(strategy)))

            print('Winner; %s' % ','.join(map(str, self.winner)))
            f_out.write('%s\n' % ','.join(map(str, self.winner)))
            f_out.close()

    def redraw_all(self):
        """
        Redraw the canvas:

         - The bike is redrawn.
         - If game is over:

           - the statistics are written to GUI

        """
        try:

            # Clear all elements
            self.canvas.tk_canvas.delete("all")
            # Draw the bike of the player
            for player in self.players:
                player.draw_bike()

            # Check if game is over
            if self.game_over:
                # Display the winner on the board
                text_str = 'Win for player %s' % str(self.winner[0])
                self.canvas.tk_canvas.create_text(100, 10, text=text_str,
                                                  font=(
                                                      "Helvetica", 12, "bold"))

        except TclError:
            pass


def main():
    """
    Setup and run the tron game.

    - Read the command-line arguments.
    - Create **STU Tron** objects
    - Run the game
    """

# If args are specified, set them
    parser = argparse.ArgumentParser()
    # Number of rows on the board
    parser.add_argument("-r", "--rows", type=int, default=40,
                        help="# of board rows")
    # Draw the board
    parser.add_argument("-d", "--draw_board", action='store_false',
                        help="draw the board or not")
    # File with an AI Player strategy
    parser.add_argument("-f", "--file", default="",
                        help="ai player strategy file")

    # Read the commandline arguments
    args = parser.parse_args()
    draw_board = args.draw_board
    rows = args.rows
    player_strategy = None
    if args.file:
        player_strategy = read_strategy_from_file(args.file)

    bike_width = 4

    # Create the game
    tron = TronAdversarial(rows=rows, bike_width=bike_width,
                           draw_board=draw_board,
                           strategy=player_strategy)
    tron.run()


if __name__ == '__main__':
    main()
