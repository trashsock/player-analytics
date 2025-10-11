import pandas as pd
import numpy as np
import json

def load_sport_config(sport='NRL'):
    """Load sport-specific configuration for AFL or NRL, including real team rosters."""
    config = {
        'NRL': {
            'field': {'length': 100, 'width': 68},
            'metrics': ['tackles', 'meters_run', 'try_assists', 'offloads'],
            'positions': ['Fullback', 'Winger', 'Centre', 'Five-eighth', 'Halfback', 'Hooker', 'Prop', 'Second-row', 'Lock'],
            'league_avg': {'tackles': 25, 'meters_run': 100, 'try_assists': 1, 'offloads': 2, 'total_distance': 6000, 'high_speed_runs': 5},
            'lineup_size': 13,
            'teams': [
                'Brisbane Broncos', 'Canberra Raiders', 'Canterbury-Bankstown Bulldogs', 'Cronulla-Sutherland Sharks', 'Dolphins', 'Gold Coast Titans', 'Manly Sea Eagles', 'Melbourne Storm', 'Newcastle Knights', 'North Queensland Cowboys', 'Parramatta Eels', 'Penrith Panthers', 'South Sydney Rabbitohs', 'St George Illawarra Dragons', 'Sydney Roosters', 'New Zealand Warriors', 'Wests Tigers'
            ],
            'rosters': {
                'Brisbane Broncos': ['Adam Reynolds', 'Ben Hunt', 'Ben Te Kura', 'Billy Walters', 'Blake Mozer', 'Brendan Piakura', 'Coby Black', 'Corey Jensen', 'Cory Paix', 'Deine Mariner', 'Delouise Hoeter', 'Ezra Mam', 'Fletcher Baker', 'Israel Leota', 'Jack Gosiewski', 'Jaiyden Hunt', 'Jesse Arthars', 'Jock Madden', 'Jordan Riki', 'Josh Rogers', 'Josiah Karapani', 'Kobe Hetherington', 'Kotoni Staggs', 'Martin Taupau', 'Patrick Carrigan', 'Payne Haas', 'Reece Walsh', 'Selwyn Cobbo', 'Tyson Smoothy', 'Xavier Willison'],
                'Canberra Raiders': ['Albert Hopoate', 'Ata Mariota', 'Chevy Stewart', 'Corey Horsburgh', 'Corey Harawira-Naera', 'Danny Levi', 'Ethan Sanders', 'Ethan Strange', 'Hudson Young', 'Jamal Fogarty', 'Joseph Tapine', 'Josh Papali\'i', 'Kaeo Weekes', 'Matt Timoko', 'Matty Nicholson', 'Michael Asomua', 'Morgan Smithies', 'Myles Martin', 'Pasami Saulo', 'Savelio Tamale', 'Sebastian Kris', 'Simi Sasagi', 'Tom Starling', 'Trey Mooney', 'Vena Patuki-Case', 'Xavier Savage', 'Zac Hosking'],
                'Canterbury-Bankstown Bulldogs': ['Bailey Hayward', 'Blake Taaffe', 'Blake Wilson', 'Bronson Xerri', 'Connor Tracey', 'Daniel Suluka-Fifita', 'Drew Hutchison', 'Enari Tuala', 'Harry Hayes', 'Jack Todd', 'Jacob Kiraz', 'Jacob Preston', 'Jaeman Salmon', 'Jake Turpin', 'Josh Curran', 'Karl Oloapu', 'Kurt Mann', 'Kurtis Morrin', 'Marcelo Montoya', 'Matt Burton', 'Max King', 'Mitchell Woods', 'Reed Mahoney', 'Ryan Sutton', 'Sam Hughes', 'Sitili Tupouniua', 'Stephen Crichton', 'Toby Sexton', 'Tom Amone', 'Viliame Kikau', 'Zyon Maiu\'u'],
                'Cronulla-Sutherland Sharks': ['Addin Fonua-Blake', 'Billy Burns', 'Blayke Brailey', 'Braden Hamlin-Uele', 'Braydon Trindall', 'Briton Nikora', 'Cameron McInnes', 'Daniel Atkinson', 'Hohepa Puru', 'Jayden Berrell', 'Jesse Colquhoun', 'Jesse Ramien', 'Kade Dykes', 'Kayal Iro', 'Liam Ison', 'Mawene Hiroti', 'Nicho Hynes', 'Niwhai Puru', 'Oregon Kaufusi', 'Ronaldo Mulitalo', 'Sam Stonestreet', 'Sione Katoa', 'Siosifa Talakai', 'Teig Wilton', 'Toby Rudolf', 'Tom Hazelton', 'Tukupa-Ke Hau Tapuha', 'William Kennedy'],
                'Dolphins': ['Connelly Lemuelu', 'Daniel Saifiti', 'Felise Kaufusi', 'Hamiso Tabuai-Fidow', 'Harrison Graham', 'Herbie Farnworth', 'Isaiya Katoa', 'Jack Bostock', 'Jake Averillo', 'Jamayne Isaako', 'Jeremiah Simbiken', 'Jeremy Marshall-King', 'Josh Kerr', 'Junior Tupou', 'Kenneath Bromwich', 'Kodi Nikorima', 'Kulikefu Finefeuiaki', 'Kurt Donoghoe', 'Mark Nicholls', 'Mason Teague', 'Max Feagai', 'Max Plath', 'Michael Waqa', 'Oryn Keeley', 'Ray Stone', 'Sean O\'Sullivan', 'Thomas Flegler', 'Tom Gilbert', 'Trai Fuller'],
                'Gold Coast Titans': ['AJ Brimson', 'Alofiana Khan-Pereira', 'Beau Fermor', 'Ben Liyou', 'Brian Kelly', 'Carter Gordon', 'Chris Randall', 'David Fifita', 'Harley Smith-Shields', 'Iszac Fa\'asuamaleaui', 'Jacob Alick-Wiencke', 'Jaimin Jolliffe', 'Jayden Campbell', 'Jaylan de Groot', 'Jojo Fifita', 'Josiah Pahulu', 'Keano Kini', 'Ken Maumalo', 'Kieran Foran', 'Klese Haas', 'Moeaki Fotuaika', 'Phillip Sami', 'Reagan Campbell-Gillard', 'Ryan Foran', 'Sam Verrills', 'Tino Fa\'asuamaleaui', 'Tony Francis'],
                'Manly Sea Eagles': ['Aaron Schoupp', 'Aitasi James', 'Ben Trbojevic', 'Clayton Faulalo', 'Daly Cherry-Evans', 'Dean Matterson', 'Ethan Bullemor', 'Gordon Chan Kum Tong', 'Haumole Olakau\'atu', 'Jake Arthur', 'Jake Simpkin', 'Jake Trbojevic', 'Jason Saab', 'Jazz Tevaga', 'Joey Walsh', 'Josh Aloiai', 'Lachlan Croker', 'Luke Brooks', 'Michael Chee Kam', 'Nathan Brown', 'Raymond Vaega', 'Reuben Garrick', 'Sio Siua Taukeiaho', 'Taniela Paseka', 'Tom Trbojevic', 'Toafofoa Sipley', 'Tolutau Koula', 'Tommy Talau'],
                'Melbourne Storm': ['Alec MacDonald', 'Ativalu Lisati', 'Bronson Garlick', 'Cameron Munster', 'Christian Welch', 'Eliesa Katoa', 'Grant Anderson', 'Harry Grant', 'Jack Howarth', 'Jahrome Hughes', 'Joe Chan', 'Jonah Pezet', 'Josh King', 'Lazarus Vaalepu', 'Moses Leo', 'Nelson Asofa-Solomona', 'Nick Meaney', 'Ryan Papenhuyzen', 'Shawn Blore', 'Stefano Utoikamanu', 'Sualauvi Fa\'alogo', 'Trent Loiero', 'Tui Kamikamica', 'Tyran Wishart', 'William Warbrick', 'Xavier Coates'],
                'Newcastle Knights': ['Adam Elliott', 'Bradman Best', 'Brodie Jones', 'Cody Hopwood', 'Dane Gagai', 'Dylan Lucas', 'Fletcher Sharpe', 'Francis Manulelieua', 'Greg Marzhew', 'Jack Cogger', 'Jack Hetherington', 'Jackson Hastings', 'Jacob Saifiti', 'James Schiller', 'Jayden Brailey', 'Jermaine McEwen', 'Kai Pearce-Paul', 'Kalyn Ponga', 'Leo Thompson', 'Mat Croker', 'Paul Bryan', 'Phoenix Crossland', 'Riley Jones', 'Sebastian Su\'a', 'Taj Annan', 'Thomas Cant', 'Tyson Frizell', 'Tyson Gamble', 'Will Pryce'],
                'North Queensland Cowboys': ['Braidon Burns', 'Coen Hess', 'Griffin Neame', 'Harrison Edwards', 'Helium Luki', 'Jake Clifford', 'Jason Taumalolo', 'Jaxon Purdue', 'Jaxson Paulo', 'Jeremiah Nanai', 'John Bateman', 'Jordan McLean', 'Kai O\'Donnell', 'Kaiden Lahrs', 'Karl Lawton', 'Murray Taulagi', 'Reece Robson', 'Reuben Cotter', 'Robert Derby', 'Sam McIntyre', 'Scott Drinkwater', 'Semi Valemei', 'Thomas Duffy', 'Thomas Mikaele', 'Tom Dearden', 'Tom Chester', 'Viliami Vailea', 'Zac Laybutt'],
                'Parramatta Eels': ['Bailey Simonsson', 'Brendan Hands', 'Bryce Cartwright', 'Dean Hawkins', 'Dylan Brown', 'Haze Dunster', 'Isaiah Iongi', 'Jack Williams', 'J\'maine Hopgood', 'Joash Papalii', 'Joey Lussick', 'Joe Ofahengaue', 'Jordan Samrani', 'Josh Addo-Carr', 'Junior Paulo', 'Kelma Tuilagi', 'Luca Moretti', 'Matt Doorey', 'Mitchell Moses', 'Ronald Volkman', 'Ryan Matterson', 'Sam Tuivaiti', 'Sean Russell', 'Shaun Lane', 'Toni Mataele', 'Will Penisini', 'Wiremu Greig', 'Zac Lomax'],
                'Penrith Panthers': ['Asu Kepaoa', 'Blaize Talagi', 'Brad Schneider', 'Brian To\'o', 'Casey McLean', 'Daine Laurie', 'Dylan Edwards', 'Harrison Hassett', 'Isaah Yeo', 'Isaiah Papali\'i', 'Izack Tago', 'Jack Cole', 'Jesse McLean', 'Liam Henry', 'Liam Martin', 'Lindsay Smith', 'Luke Garner', 'Luke Sommerton', 'Luron Patea', 'Mavrik Geyer', 'Matt Eisenhuth', 'Mitch Kenny', 'Moses Leota', 'Nathan Cleary', 'Paul Alamoti', 'Preston Riki', 'Riley Price', 'Scott Sorensen', 'Soni Luke'],
                'South Sydney Rabbitohs': ['Alex Johnston', 'Ben Lovett', 'Cameron Murray', 'Campbell Graham', 'Cody Walker', 'Davvy Moale', 'Euan Aitken', 'Haizyn Mellars', 'Isaiah Tass', 'Jack Wighton', 'Jacob Host', 'Jai Arrow', 'Jamie Humphreys', 'Jayden Sullivan', 'Josh Schuster', 'Jye Gray', 'Keaon Koloamatangi', 'Lachlan Hubner', 'Latrell Mitchell', 'Lewis Dodd', 'Peter Mamouzelos', 'Sean Keppie', 'Shaquai Mitchell', 'Siliva Havili', 'Tallis Duncan', 'Tevita Tatola', 'Thomas Fletcher', 'Tyrone Munro'],
                'St George Illawarra Dragons': ['Blake Lawrie', 'Christian Tuipulotu', 'Clint Gutherson', 'Cody Ramsey', 'Corey Allan', 'Damien Cook', 'Emre Guler', 'Francis Molo', 'Hame Sele', 'Hamish Stewart', 'Jack de Belin', 'Jacob Liddle', 'Jaydn Su\'A', 'Kyle Flanagan', 'Lachlan Ilias', 'Luciano Leilua', 'Mathew Feagai', 'Michael Molo', 'Mikaele Ravalawa', 'Moses Suli', 'Raymond Faitala-Mariner', 'Ryan Couchman', 'Sione Finau', 'Toby Couchman', 'Tom Eisenhuth', 'Tyrell Sloan', 'Valentine Holmes', 'Viliami Fifita'],
                'Sydney Roosters': ['Angus Crichton', 'Billy Smith', 'Blake Steep', 'Brandon Smith', 'Chad Townsend', 'Connor Watson', 'Daniel Tupou', 'De La Salle Va\'a', 'Dominic Young', 'Egan Butcher', 'James Tedesco', 'Junior Pauga', 'Lindsay Collins', 'Mark Nawaqanitawase', 'Nat Butcher', 'Naufahu Whyte', 'Robert Toia', 'Sam Walker', 'Sandon Smith', 'Siua Wong', 'Spencer Leniu', 'Toby Rodwell', 'Tyler Moriarty', 'Victor Radley'],
                'New Zealand Warriors': ['Adam Pompey', 'Ali Leiataua', 'Bunty Afoa', 'Chanel Harris-Tavita', 'Charnze Nicoll-Klokstad', 'Dallin Watene-Zelezniak', 'Demitric Vaimauga', 'Dylan Walker', 'Edward Kosi', 'Erin Clark', 'Freddy Lussick', 'Jackson Ford', 'Jacob Laban', 'James Fisher-Harris', 'Kurt Capewell', 'Luke Metcalf', 'Marata Niukore', 'Mitchell Barnett', 'Motu Pasikala', 'Rocco Berry', 'Roger Tuivasa-Sheck', 'Sam Healey', 'Selumiela Halasima', 'Taine Tuaupiki', 'Tanah Boyd', 'Te Maire Martin', 'Tom Ale', 'Wayde Egan'],
                'Wests Tigers': ['Adam Doueihi', 'Alex Twal', 'Alex Seyfarth', 'Apisai Koroisau', 'Brandon Tumeth', 'Brent Naden', 'Charlie Staines', 'Chris Faagatu', 'David Klemmer', 'Fonua Pole', 'Jack Bird', 'Jahream Bula', 'Jarome Luai', 'Jeral Skelton', 'Jordan Miller', 'Josh Feledy', 'Justin Matamua', 'Justin Olam', 'Kit Laulilii', 'Lachlan Galvin', 'Latu Fainu', 'Luke Laulilii', 'Royce Hunt', 'Samuela Fainu', 'Sione Fainu', 'Solomone Saukuru', 'Starford To\'a', 'Sunia Turuva', 'Tallyn Da Silva', 'Terrell May', 'Tony Sukkar']
            }
        },
        'AFL': {
            'field': {'length': 185, 'width': 155},
            'metrics': ['disposals', 'marks', 'clearances', 'contested_possessions'],
            'positions': ['Midfielder', 'Forward', 'Defender', 'Ruck', 'Wing'],
            'league_avg': {'disposals': 20, 'marks': 5, 'clearances': 3, 'contested_possessions': 8, 'total_distance': 12000, 'high_speed_runs': 10},
            'lineup_size': 18,
            'teams': ['Adelaide Crows', 'Brisbane Lions', 'Carlton Blues', 'Collingwood Magpies', 'Essendon Bombers', 'Fremantle Dockers', 'Geelong Cats', 'Gold Coast Suns', 'GWS Giants', 'Hawthorn Hawks', 'Melbourne Demons', 'North Melbourne Kangaroos', 'Port Adelaide Power', 'Richmond Tigers', 'St Kilda Saints', 'Sydney Swans', 'West Coast Eagles', 'Western Bulldogs'],
            'rosters': {
                'Adelaide Crows': ['Taylor Walker', 'Jordan Dawson', 'Rory Laird', 'Ben Keays', 'Reilly O\'Brien', 'Izak Rankine', 'Darcy Fogarty', 'Lachlan Sholl', 'Matt Crouch', 'Josh Rachele', 'Ned McHenry', 'Riley Thilthorpe', 'Brodie Smith', 'Jake Soligo', 'Lachlan Murphy', 'Max Michalanney', 'Luke Nankervis', 'Josh Worrell', 'Billy Dowling', 'Mark Keane', 'James Borlase', 'Kieran Strachan', 'Luke Pedlar', 'Brayden Cook', 'Chayce Jones', 'Christopher Burgess', 'Daniel Curtin', 'Harry Schoenberg', 'Karl Gallagher', 'Nicholas Murray', 'Oscar Ryan', 'Toby Murray', 'Will Hamill', 'Zac Taylor'],
                'Brisbane Lions': ['Joe Daniher', 'Lachie Neale', 'Hugh McCluggage', 'Josh Dunkley', 'Harris Andrews', 'Dayne Zorko', 'Charlie Cameron', 'Cameron Rayner', 'Jarrod Berry', 'Darcy Wilmot', 'Callum Ah Chee', 'Zac Bailey', 'Keidean Coleman', 'Jack Payne', 'Jaspa Fletcher', 'Oscar McInerney', 'Eric Hipwood', 'Brandon Starcevich', 'Kai Lohmann', 'Ryan Lester', 'Logan Morris', 'Noah Answerth', 'Conor McKenna', 'Darcy Gardiner', 'Tom Doedee', 'Jaxon Prior', 'Harry Sharp', 'Henry Smith', 'Luke Lloyd', 'Bruce Reville', 'Darragh Joyce', 'Shadeau Brain', 'Will McLachlan', 'Jonty Scharenberg', 'Darcy Craven'],
                # (Truncated for brevity)
                'Carlton Blues': ['Patrick Cripps', 'Sam Walsh', 'Harry McKay', 'Charlie Curnow', 'Jacob Weitering', 'Adam Cerra', 'Tom De Koning', 'Nic Newman', 'Blake Acres', 'Matthew Kennedy', 'George Hewett', 'Adam Saad', 'Jack Martin', 'Matthew Cottrell', 'Orazio Fantasia', 'Mitch McGovern', 'Jordan Boyd', 'Jesse Motlop', 'Lachie Fogarty', 'Zac Williams', 'Caleb Marchbank', 'Jack Carroll', 'David Cuningham', 'Oliver Hollands', 'Alex Cincotta', 'Alex Mirkov', 'Ashton Moir', 'Billy Wilson', 'Cooper Lord', 'Corey Durdin', 'Harry Lemmey', 'Hudson O\'Keeffe', 'Jaxon Binns', 'Liam McMahon', 'Matthew Owies'],
                'Collingwood Magpies': ['Scott Pendlebury', 'Nick Daicos', 'Jordan De Goey', 'Josh Daicos', 'Jamie Elliott', 'Darcy Moore', 'Jack Crisp', 'Brayden Maynard', 'Steele Sidebottom', 'John Noble', 'Pat Lipinski', 'Darcy Cameron', 'Brody Mihocek', 'Isaac Quaynor', 'Beau McCreery', 'Bobby Hill', 'Billy Frampton', 'Jeremy Howe', 'Joe Richards', 'Harvey Harrison', 'Reef McInnes', 'Lachlan Sullivan', 'Ned Long', 'Charlie Dean', 'Wil Parker', 'Edward Allan', 'Finlay Macrae', 'Ash Johnson', 'Jakob Ryan', 'Tew Jiath', 'Harry DeMattia', 'Jack Bytel'],
                'Essendon Bombers': ['Zach Merrett', 'Dyson Heppell', 'Darcy Parish', 'Jake Stringer', 'Kyle Langford', 'Jye Caldwell', 'Nic Martin', 'Sam Durham', 'Andrew McGrath', 'Jordan Ridley', 'Mason Redman', 'Peter Wright', 'Harrison Jones', 'Archie Perkins', 'Ben McKay', 'Jayden Laverde', 'Jake Kelly', 'Xavier Duursma', 'Todd Goldstein', 'Elijah Tsatas', 'Nate Caddy', 'Vigo Visentini', 'Luamon Lual', 'Saad El-Hawli', 'Tex Wanganeen', 'Billy Cootee', 'Kaine Baldwin', 'Matt Guelfi', 'Nick Bryan', 'Nick Hind', 'Dylan Shiel', 'Jade Gresham'],
                'Fremantle Dockers': ['Luke Jackson', 'Andrew Brayshaw', 'Caleb Serong', 'Hayden Young', 'Nat Fyfe', 'Sean Darcy', 'Josh Treacy', 'Jye Amiss', 'Luke Ryan', 'Jordan Clark', 'Michael Walters', 'Matthew Johnson', 'Jaeger O\'Meara', 'Tom Emmett', 'Brandon Walker', 'Neil Erasmus', 'Ethan Hughes', 'Karl Worner', 'Sam Sturt', 'James Aish', 'Corey Wagner', 'Patrick Voss', 'Liam Reidy', 'Cooper Simpson', 'Will Brodie', 'Heath Chapman', 'Josh Draper', 'Hugh Davies', 'Josh Corbett', 'Alex Pearce', 'Bailey Banfield', 'Michael Frederick', 'Sam Switkowski', 'Shai Bolton'],
                'Geelong Cats': ['Patrick Dangerfield', 'Tom Hawkins', 'Jeremy Cameron', 'Mitch Duncan', 'Tom Stewart', 'Mark Blicavs', 'Jack Henry', 'Max Holmes', 'Gryan Miers', 'Tanner Bruhn', 'Jhye Clark', 'Zach Guthrie', 'Ollie Dempsey', 'Oisin Mullin', 'Shannon Neale', 'Sam De Koning', 'Jake Kolodjashnij', 'Zach Tuohy', 'Rhys Stanley', 'Brad Close', 'Jack Bowes', 'Ollie Henry', 'Lawson Humphries', 'Connor O\'Sullivan', 'Phoenix Foster', 'Mitch Edwards', 'Shaun Mannagh', 'Mitch Hardie', 'George Stevens', 'Joe Furphy', 'Oliver Wiltshire'],
                'Gold Coast Suns': ['Matt Rowell', 'Noah Anderson', 'Sam Flanders', 'Ben King', 'Touks Miller', 'Jack Lukosius', 'Ben Ainsworth', 'Ben Long', 'Bodhi Uwland', 'Mac Andrew', 'Sam Collins', 'Alex Davies', 'Will Graham', 'Jed Walter', 'Ethan Read', 'Bailey Humphrey', 'Nick Holman', 'Jarrod Witts', 'David Swallow', 'Malcolm Rosas', 'Jake Rogers', 'Joel Jeffrey', 'Sam Clohesy', 'Lloyd Johnston', 'Will Rowlands', 'Ned Moyle', 'Thomas Berry', 'Connor Budarick', 'Alex Sexton', 'Darcy Macpherson', 'Brayden Fiorini', 'Mal Rosas'],
                'GWS Giants': ['Jesse Hogan', 'Tom Green', 'Josh Kelly', 'Stephen Coniglio', 'Lachie Whitfield', 'Toby Greene', 'Callan Ward', 'Harry Perryman', 'Sam Taylor', 'Lachie Ash', 'Harry Himmelberg', 'Jake Riccardi', 'Callum Brown', 'Finn Callaghan', 'Aaron Cadman', 'Darcy Jones', 'Harvey Thomas', 'Jacob Wehr', 'Toby Bedford', 'Toby McMullin', 'Leek Aleer', 'Conor Stone', 'Phoenix Gothard', 'James Leake', 'Max Gruzewski', 'Joe Fonti', 'Jack Steele', 'Lachlan Keeffe', 'Ryan Angwin', 'Nick Haynes', 'Isaac Cumming', 'Harry Rowston'],
                'Hawthorn Hawks': ['James Sicily', 'Blake Hardwick', 'Luke Breust', 'Jack Scrimshaw', 'Karl Amon', 'Jai Newcombe', 'James Worpel', 'Will Day', 'Dylan Moore', 'Jack Ginnivan', 'Conor Nash', 'Mabior Chol', 'Ned Reeves', 'Changkuoth Jiath', 'Harry Morrison', 'Josh Ward', 'Josh Weddle', 'Finn Maginness', 'Massimo D\'Ambrosio', 'Seamus Mitchell', 'Connor Macdonald', 'Calsher Dear', 'Nick Watson', 'Henry Hustwaite', 'Bodhi Uwland', 'Cam Mackenzie', 'Harry Dear', 'Jasper Scaife', 'Cooper Stephens', 'Jack O\'Sullivan', 'Clay Tucker', 'Luke Bruest', 'Ethan Phillips'],
                'Melbourne Demons': ['Max Gawn', 'Christian Petracca', 'Jack Viney', 'Clayton Oliver', 'Steven May', 'Jake Lever', 'Ed Langdon', 'Kysaiah Pickett', 'Bayley Fritsch', 'Tom Sparrow', 'Harrison Petty', 'Alex Neal-Bullen', 'Christian Salem', 'Judd McVee', 'Tom McDonald', 'Jacob van Rooyen', 'Caleb Windsor', 'Koltyn Tholstrup', 'Blake Howes', 'Kade Chandler', 'Andy Moniz-Wakefield', 'Shane McAdam', 'Marty Hore', 'Ben Brown', 'Josh Schache', 'Matthew Jefferson', 'Will Verrall', 'Daniel Turner', 'Adam Tomlinson', 'Joel Smith', 'Bailey Laurie', 'Lachie Hunter', 'Kyah Farris-White', 'Oliver Sestan'],
                'North Melbourne Kangaroos': ['Luke Davies-Uniacke', 'Harry Sheezel', 'Nick Larkey', 'Tristan Xerri', 'Luke McDonald', 'Zane Duursma', 'Tom Powell', 'George Wardlaw', 'Colby McKercher', 'Will Phillips', 'Jy Simpkin', 'Aidan Corr', 'Callum Coleman-Jones', 'Eddie Ford', 'Griffin Logue', 'Charlie Comben', 'Jackson Archer', 'Liam Shiels', 'Riley Hardeman', 'Brayden George', 'Curtis Taylor', 'Dylan Stephens', 'Paul Curtis', 'Toby Pink', 'Tyler Sellers', 'Charlie Lazzaro', 'Blake Drury', 'Cooper Harvey', 'Miller Bergman', 'Robert Hansen Jr', 'Wil Dawson', 'Zac Fisher'],
                'Port Adelaide Power': ['Connor Rozee', 'Zak Butters', 'Jason Horne-Francis', 'Charlie Dixon', 'Dan Houston', 'Darcy Byrne-Jones', 'Kane Farrell', 'Aliir Aliir', 'Willem Drew', 'Todd Marshall', 'Travis Boak', 'Mitch Georgiades', 'Ryan Burton', 'Quinton Narkle', 'Francis Evans', 'Ollie Wines', 'Jase Burgoyne', 'Dylan Williams', 'Miles Bergman', 'Josh Sinn', 'Lachie Jones', 'Ivan Soldo', 'Jordon Sweet', 'Will Lorenz', 'Tom Clurey', 'Esava Ratugolea', 'Jed McEntee', 'Ollie Lord', 'Dante Visentini', 'Jeremy Finlayson', 'Hugh Jackson', 'Tom Scully', 'Jack Hayes', 'Willie Rioli'],
                'Richmond Tigers': ['Dustin Martin', 'Tom Lynch', 'Shai Bolton', 'Jayden Short', 'Liam Baker', 'Daniel Rioli', 'Tim Taranto', 'Dion Prestia', 'Jacob Hopper', 'Nick Vlastuin', 'Jack Graham', 'Nathan Broad', 'Noah Balta', 'Samson Ryan', 'Sam Banks', 'Seth Campbell', 'Kane McAuliffe', 'Matthew Coulthard', 'Jacob Bauer', 'Steely Green', 'Josh Gibcus', 'Mykelti Lefau', 'Tylar Young', 'Judson Clarke', 'James Trezise', 'Tom Brown', 'Noah Cumberland', 'Jacob Koschitzke', 'Liam Fawcett', 'Kaleb Smith', 'Campbell Gray', 'Luke Maddock', 'Harrison White'],
                'St Kilda Saints': ['Jack Steele', 'Bradley Hill', 'Nasiah Wanganeen-Milera', 'Rowan Marshall', 'Callum Wilkie', 'Jack Sinclair', 'Darcy Wilson', 'Mattaes Phillipou', 'Josh Battle', 'Mitch Owens', 'Max King', 'Liam Henry', 'Liam Stocker', 'Jack Higgins', 'Marcus Windhager', 'Bradley Crouch', 'Mason Wood', 'Cooper Sharman', 'Sebastian Ross', 'Zaine Cordy', 'Zak Jones', 'Paddy Dow', 'Riley Bonner', 'Tom Campbell', 'Hunter Clark', 'Arie Schoenmaker', 'Angus Hastie', 'James Van Es', 'Isaac Keeler', 'Jack Hayes', 'Olli Hotton', 'Lance Collard'],
                'Sydney Swans': ['Isaac Heeney', 'Chad Warner', 'Errol Gulden', 'Tom Papley', 'Nick Blakey', 'Brodie Grundy', 'Luke Parker', 'Taylor Adams', 'Joel Amartey', 'James Rowbottom', 'Oliver Florent', 'Dane Rampe', 'Harry Cunningham', 'Will Hayward', 'Logan McDonald', 'Jake Lloyd', 'Robbie Fox', 'Matt Roberts', 'James Jordon', 'Braeden Campbell', 'Lewis Melican', 'Caiden Cleary', 'Indhi Kirk', 'Aaron Francis', 'Sam Wicks', 'Peter Ladhams', 'Caleb Mitchell', 'Cooper Vickery', 'Joel Hamling', 'Jaiden Magor', 'Will Edwards', 'Lachlan McAndrew', 'Angus Sheldrick'],
                'West Coast Eagles': ['Jake Waterman', 'Tom Barrass', 'Elliot Yeo', 'Tim Kelly', 'Liam Duggan', 'Andrew Gaff', 'Jeremy McGovern', 'Harley Reid', 'Oscar Allen', 'Reuben Ginbey', 'Jayden Hunt', 'Jack Darling', 'Jamie Cripps', 'Liam Ryan', 'Bailey Williams', 'Campbell Chesser', 'Clay Hall', 'Jack Petruccelle', 'Jack Williams', 'Ryan Maric', 'Archer Reid', 'Harry Barnett', 'Harvey Johnston', 'Jack Hutchinson', 'Tyrell Dewar', 'Callum Jamieson', 'Harry Edwards', 'Jordyn Baker', 'Alex Witherden', 'Matt Flynn', 'Zane Trew', 'Coby Burgiel'],
                'Western Bulldogs': ['Marcus Bontempelli', 'Tom Liberatore', 'Adam Treloar', 'Aaron Naughton', 'Jack Macrae', 'Ed Richards', 'Bailey Dale', 'Liam Jones', 'Caleb Daniel', 'Bailey Smith', 'Jamarra Ugle-Hagan', 'Sam Darcy', 'Tim English', 'Rory Lobb', 'Rhylee West', 'Lachlan McNeil', 'James O\'Donnell', 'Harvey Gallagher', 'Arthur Jones', 'Joel Freijah', 'Lachlan Bramble', 'Luke Cleary', 'Oskar Baker', 'Riley Garcia', 'Jordan Croft', 'Dominic Bedendo', 'Charlie Clarke', 'Jed Hagan', 'Alex Keath', 'Nick Coffield', 'Laitham Vandermeer', 'Cody Weightman']
            }  # Sample AFL rosters
        }
    }
    return config[sport]

def generate_league_data(sport='NRL', n_games=10, roster_file=None):
    """Generate synthetic match and GPS data for ALL teams in AFL or NRL."""
    config = load_sport_config(sport)
    all_main = []
    all_gps = []
    
    for team_name in config['teams']:
        # Roster from config or CSV
        if roster_file:
            roster = pd.read_csv(roster_file)
            players = roster['player'].tolist()
            positions = roster['position'].tolist()
            ages = roster.get('age', np.random.randint(18, 35, len(players))).astype('int8')
        else:
            players = config['rosters'].get(team_name, [f'{team_name}_Player_{i}' for i in range(30)])
            positions = np.random.choice(config['positions'], len(players)).tolist()
            ages = np.random.randint(18, 35, len(players)).astype('int8')
            ages[:5] = np.random.randint(18, 23, 5)  # Young players

        n_players = len(players)
        # Main dataset per team
        np.random.seed(42 + hash(team_name) % (2**32 - 1))
        data = {
            'team': [team_name] * n_players * n_games,
            'player': np.repeat(players, n_games),
            'position': np.repeat(positions, n_games),
            'age': np.repeat(ages, n_games),
            'game_date': pd.date_range('2025-03-01', '2025-10-10', periods=n_games).repeat(n_players),
            'fatigue_score': np.random.uniform(0, 100, n_players * n_games).astype('float32'),
            'injury_status': np.random.choice([0, 1], n_players * n_games, p=[0.9, 0.1]).astype('float32')
        }
        for metric in config['metrics']:
            data[metric] = np.random.randint(0, 50, n_players * n_games).astype('float32')
        df_main_team = pd.DataFrame(data)
        all_main.append(df_main_team)

        # GPS data per team
        n_time_points = 10
        gps_data = {
            'team': [team_name] * n_players * n_games * n_time_points,
            'player': np.repeat(df_main_team['player'], n_time_points),
            'game_date': np.repeat(df_main_team['game_date'], n_time_points),
            'timestamp': np.tile(np.arange(n_time_points), len(df_main_team)),
            'x_pos': np.random.uniform(0, config['field']['length'], len(df_main_team) * n_time_points).astype('float32'),
            'y_pos': np.random.uniform(0, config['field']['width'], len(df_main_team) * n_time_points).astype('float32'),
            'speed': np.random.uniform(0, 10, len(df_main_team) * n_time_points).astype('float32'),
            'acceleration': np.random.uniform(-5, 5, len(df_main_team) * n_time_points).astype('float32')
        }
        df_gps_team = pd.DataFrame(gps_data)
        all_gps.append(df_gps_team)

        # Aggregate GPS for main
        gps_agg = df_gps_team.groupby(['team', 'player', 'game_date']).agg(
            total_distance=('speed', lambda x: np.sum(x) * 8),
            high_speed_runs=('speed', lambda x: np.sum(x > 5)),
            max_acceleration=('acceleration', 'max')
        ).reset_index()
        df_main_team = df_main_team.merge(gps_agg, on=['team', 'player', 'game_date'])
        all_main[-1] = df_main_team

    # Combine league-wide
    df_main = pd.concat(all_main, ignore_index=True)
    df_gps = pd.concat(all_gps, ignore_index=True)

    # Save
    print("Saving stats: ")
    df_main.to_csv(f'league_stats_{sport}.csv', index=False)
    df_gps.to_csv(f'league_gps_raw_{sport}.csv', index=False)
    with open(f'sport_config_{sport}.json', 'w') as f:
        json.dump(config, f)
    return df_main, df_gps, config
