projects = [
    {
        "id" : 1,
        "title": "Corpers Empire",
        "overview": `Corpers Empire is a web-based application which enables National Youth Service Corps
                    Members in Oyo state to find suitable accomodation for their service year with ease.
                    It serves as a platform connecting corp members to available accommodations without having
                    to pass through the expensive process of using agents.`,
        "features": [
                     "The user can surf through a catalogue of accommodations",
                     "They Can also search for accomodation using an advance filter",
                     "They can register interest on the accomodation of their choice",
                     "The admin can add, update and delete accomodations",
                     "The admin can view all lists of pending interests",
                     "The admin can control the status of an accomodation"
                    ],
        "Tools": [
                    "HTML", "CSS", "Bootstrap", "Vanilla Javascript", "Jquery", "PHP", "MySQL"
                 ],
        "small_screen" : "1sa.jpg",
        "all_images" : ['1a.jpg', '1b.jpg', '1c.jpg', '1d.jpg', '1e.jpg', '1f.jpg', '1g.jpg', '1h.jpg', '1i.jpg'],
        "pages":[
                    [
                        "The Landing Page",
                        `The top section of the landing page displays animated description of the website.
                        Its mid section displays images and little overview of the most preffered kinds of 
                        appartment. While the last section contains the vision statement`,
                        ['1a.jpg', '1b.jpg', '1c.jpg']
                    ],
                    [
                        "The Catalogue Page",
                        `The catalogue page displays all available accomodations with filter to search for 
                        specific types, amount, area etc.`,
                        ['1d.jpg']
                    ],
                    [
                        'The Apartment Description Page',
                        `The appartment description page provides indept description of the apartment.
                        Its features, cost, images, area etc. It also provides a button to indicate interest.`,
                        ['1e.jpg', '1f.jpg']
                    ],
                    [
                        'The Chechout Page',
                        `The checkout page provides a form for collecting details of interested users.`,
                        ['1g.jpg']
                    ],
                    [
                        'The Admin Page',
                        `The admin page provides buttons to features such as adding, updating and deleting accomodations; 
                        viewing pending interests on accomodation and controlling the status of accomodations`,
                        ['1h.jpg', '1i.jpg']
                    ]
        ]
    },

    {
        "id" : 2,
        "title": "Electronic Examination Software",
        "overview": `An electronic examinations software for primary school pupil. It was designed with simple interface
                    to allow accessibility for the young targetted users. It provides features for authoring and editing questions using 
                    files and enabling and disabling tests. It also maintains states of progress even after page refresh.`,
        "features": [
                     "Pupils can Login using their Name and Examination number",
                     "They can select from the currently active tests which tests to take",
                     "They can proceed to take their tests",
                     "The admin can add tests, upload questions and upload participating pupils",
                     "The admin can view tests and activate tests before the start time or deactivates test after the schedule",
                     "The admin can print hard copy of questions and tests scores"
                    ],
        "Tools": [
                    "HTML", "CSS", "Bootstrap", "Vanilla Javascript", "Jquery", "PHP", "MySQL"
                 ],
        "small_screen" : "2sa.jpg",
        "all_images" : ['2a.jpg', '2b.jpg', '2c.jpg', '2d.jpg', '2e.jpg', '2f.jpg'],
        "pages":[
                    [
                        "The Pupil Login Page",
                        `This page provides a login form for pupils to enter their name and examination number.
                        Through this page, the pupil can continue to select from the active tests to participate in`,
                        ['2a.jpg', '2b.jpg']
                    ],
                    [
                        "The Test Page",
                        `This page has several features which makes taking the examination fun and stress free.`,
                        ['2c.jpg']
                    ],
                    [
                        'The Admin Page',
                        `The Admin page has features which allows a central administrator to create a test, upload questions
                        to the test, and add participating pupils' details to the test. The admin can also enable and disable
                        tests. Through this page, the admin can also print hard copies of the questions and test results`,
                        ['2d.jpg', '2e.jpg', '2f.jpg']
                    ],
                ]
    },

    {
        "id" : 3,
        "title": "Minlands Pro",
        "overview": `An application for digitizing land ownership records in Nigeria. It is intended to serve as first 
                    point of reference for confirming previous records on piece of lands before C of O's are tendered.
                    This is to reduce the rate of conflicts on land matters `,
        "features": [
                     "The Surveyor Registers new surveyors",
                     "Registered Surveyors can add Title Deed Plans",
                     "Registered Surveyors can keep track of their TDPs",
                     "The HOD cartography is able to view submitted TDP on maps provided in the application",
                     "The HOD is able to accept or reject the maps depending if there are no conflicting interest",
                     "Accepted TDPs are forwarded to the Assistant Surveyor general",
                     "The surveyor General is also able to check the TDP and accept or reject the TDP",
                     "The Accepted TDP is forwarded to the surveyor General who makes final acceptance or rejection"
                    ],
        "Tools": [
                    "HTML", "CSS", "Bootstrap", "Vanilla Javascript", "Jquery", "PHP", "MySQL"
                 ],
        "small_screen" : "3b.jpg",
        "all_images" : ['3a.jpg', '3b.jpg', '3c.jpg', '3d.jpg', '3e.jpg', '3f.jpg', '3g.jpg'],
        "pages":[
                    [
                        "The Landing Page",
                        `This page provides links to portals of each user; the surveyor general, the HOD Cartography, 
                        The Deputy Surveyor General and the Surveyor`,
                        ['3a.jpg']
                    ],
                    [
                        "The Login Pages",
                        `Each category of user have their distinct login pages`,
                        ['3b.jpg']
                    ],
                    [
                        'The Surveyor General Page',
                        `The Surveyor General page encopases features which enables the surveyor general to
                        add surveyors, view registered surveyors, view submitted TDP and accept or reject TDPs`,
                        ['3c.jpg', '3d.jpg']
                    ],
                    [
                        'The Deputy Surveyor General Page',
                        `The Deputy surveyor general page displays queues of pending TDPs. It has feature which allows the
                        DSG to double check TDP and accept or reject them`,
                        []
                    ],

                    [
                        'The HOD Cartography Page',
                        `The HOD cartography page enables the HOD cartography to view details of TDP and view land on
                        the application map and provided buttons to accept or rejects land`,
                        ['3g.jpg']
                    ],
                    [
                        'The Surveyor Page',
                        `This page allows surveyors to add TDP and track the status`,
                        ['3g.jpg']
                    ],
                ]
    },

    {
        "id" : 4,
        "title": "Medical Video Mergin Application",
        "overview": `A web-based application developed during the Covid pandemic to enable medical doctors create documentary 
                    about Covid remotely. It allows a central user to create passcodes for doctors to append their short videos 
                    to the already existing video.`,
        "features": [
                     "The Administrator can create a passcode and asign maximum numbers of videos to be attached by the passcoede user",
                     "The Administrator can upload base video on which other users will append",
                     "Each person authorised with a passcode can login with their passcodes and provide their proffessional details",
                     "Each person authorised with a passcode can record and append video to the base videos",
                     "The appended video can be shared to various social media networks like facebook, watsapp, instagram, twitter etc",
                    ],
        "Tools": [
                    "HTML", "CSS", "Bootstrap", "Vanilla Javascript", "Jquery", "PHP", "MySQL"
                 ],
        "small_screen" : "4sa.jpg",
        "all_images" : ['4a.jpg', '4b.jpg', '4c.jpg', '4d.jpg', '4e.jpg', '4f.jpg', '4g.jpg', '4h.jpg'],
        "pages":[
                    [
                        "The Admin Page",
                        `The admin portal provides the system administrator with the capacity to 
                        To create passcodes and determine how many uploads are allowed per created passcode. 
                        He also assigns base videos to passcodes and tracks all videos appends., 
                        `,
                        ['4a.jpg', '4b.jpg', '4c.jpg', '4d.jpg']
                    ],
                    [
                        "The User Logins",
                        `This page allows users to login with assigned passcode. The users can also enter their proffessional information 
                        post login to brand their parts of the video with their details`,
                        ['4e.jpg', '4f.jpg']
                    ],
                    [
                        'The Video capturing and Mergin Page',
                        `Through this page, the authorised user can record or select video and merge to the base video`,
                        ['4g.jpg', '4h.jpg']
                    ],
                    [
                        'The Deputy Surveyor General Page',
                        `The Deputy surveyor general page displays queues of pending TDPs. It has feature which allows the
                        DSG to double check TDP and accept or reject them`,
                        []
                    ],

                    [
                        'The HOD Cartography Page',
                        `The HOD cartography page enables the HOD cartography to view details of TDP and view land on
                        the application map and provided buttons to accept or rejects land`,
                        ['3g.jpg']
                    ],
                    [
                        'The Surveyor Page',
                        `This page allows surveyors to add TDP and track the status`,
                        ['3g.jpg']
                    ],
                ]
    },

    {
        "id" : 5,
        "title": "Minna Medium Security Prison Information System",
        "overview": `This web-based platform was designed to digitize prison Information system in Nigeria.
                    The system contains information about status of inmates, their bio, crime, court sentence, roll call, cell 
                    information etc. It also has provides an API for organisations to confirm the criminal records of individuals
                    easily.`,
        "features": [
                     "The Administrator can Register an Inmate",
                     "The Administrator can view inmates records",
                     "The Administrator can take inmate rollcall",
                     "The Administrator can bail inmates",
                     "The Administrator can declear inmates missing",
                     "The public user can search inmates using their National Identity Number",
                     "Developers can consume the API for verifying criminal records"
                    ],
        "Tools": [
                    "HTML", "CSS", "Bootstrap", "Vanilla Javascript", "Jquery", "PHP", "MySQL"
                 ],
        "small_screen" : "5sa.jpg",
        "all_images" : ['5a.jpg', '5b.jpg', '5c.jpg', '5d.jpg', '5e.jpg', '5f.jpg', '5g.jpg', '5h.jpg'],
        "pages":[
                    [
                        "The Admin Page",
                        `The admin portal provides the system administrator with the capacity to 
                        To create passcodes and determine how many uploads are allowed per created passcode. 
                        He also assigns base videos to passcodes and tracks all videos appends., 
                        `,
                        ['4a.jpg', '4b.jpg', '4c.jpg', '4d.jpg']
                    ],
                    [
                        "The User Logins",
                        `This page allows users to login with assigned passcode. The users can also enter their proffessional information 
                        post login to brand their parts of the video with their details`,
                        ['4e.jpg', '4f.jpg']
                    ],
                    [
                        'The Video capturing and Mergin Page',
                        `Through this page, the authorised user can record or select video and merge to the base video`,
                        ['4g.jpg', '4h.jpg']
                    ],
                    [
                        'The Deputy Surveyor General Page',
                        `The Deputy surveyor general page displays queues of pending TDPs. It has feature which allows the
                        DSG to double check TDP and accept or reject them`,
                        []
                    ],

                    [
                        'The HOD Cartography Page',
                        `The HOD cartography page enables the HOD cartography to view details of TDP and view land on
                        the application map and provided buttons to accept or rejects land`,
                        ['3g.jpg']
                    ],
                    [
                        'The Surveyor Page',
                        `This page allows surveyors to add TDP and track the status`,
                        ['3g.jpg']
                    ],
                ]
    }
]
