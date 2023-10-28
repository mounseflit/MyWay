import Link from "next/link";
import '../styles/main.css'
import Image from "next/image";


export default function HomePage() {
  
  const mobilePage = (
    <div className="device-notification">
      <a className="device-notification--logo" href="index.html">
        <Image width="16" height="16" layout="responsive" src="/img/logo.png" alt="RoadMapper" />
        <p>My Way &nbsp; طريقي</p>
      </a>
      <p className="device-notification--message">
        My Way have so much to offer that we must request you orient your device
        to portrait or find a larger screen. You won't be disappointed.
      </p>
    </div>
  );

  const mainPage = (
    <div className="perspective effect-rotate-left">
      <div className="container">
        <div className="outer-nav--return"></div>

        <div id="viewport" className="l-viewport">
          <div className="l-wrapper">
            <header className="header">
              <a className="header--logo" href="#home">
                <Image width="16" height="16" layout="responsive" src="/img/logo.png" alt="RoadMapper" />
                <p>My Way &nbsp; طريقي</p>
              </a>
              <a
                href="login.html"
                className="header--cta cta is-active"
                id="try-it-button"
              >
                LogIn
              </a>
              <div
       
                style={{all: 'unset'}}
              ></div>
              <script src="//embed.typeform.com/next/embed.js"></script>

              <div className="header--nav-toggle">
                <span></span>
              </div>
            </header>

            <nav className="l-side-nav">
              <ul className="side-nav">
                <li className="is-active">
                  <span>Home</span>
                </li>
                <li>
                  <span>About</span>
                </li>
                <li>
                  <span>Try It</span>
                </li>
                <li>
                  <span>Us</span>
                </li>
                <li>
                  <span>Contact</span>
                </li>
              </ul>
            </nav>

            <ul className="l-main-content main-content">
              <li className="l-section section section--is-active">
                <div className="intro">
                  <div className="intro--banner" id="home">
                    <h1>
                      Your <br /> Personalized
                      <br />
                      Path to
                      <span
                        style={{
                          color: "rgb(127, 0, 245)",
                          textShadow: "0 0px 50px rgba(54, 0, 90, 0.767)",
                        }}
                      >
                        Success
                      </span>
                    </h1>
                    <button id="showme-button">
                      Try Now...
                      <svg
                        version="1.1"
                        id="Layer_1"
                        
                        
                        x="0px"
                        y="0px"
                        viewBox="0 0 150 118"
                        
                      >
                        <g transform="translate(0.000000, 118.000000) scale(0.100000,-0.100000)">
                          <path d="M870,1167c-34-17-55-57-46-90c3-15,81-100,194-211l187-185l-565-1c-431,0-571-3-590-13c-55-28-64-94-18-137c21-20,33-20,597-20h575l-192-193C800,103,794,94,849,39c20-20,39-29,61-29c28,0,63,30,298,262c147,144,272,271,279,282c30,51,23,60-219,304C947,1180,926,1196,870,1167z" />
                        </g>
                      </svg>
                      <span className="btn-background"></span>
                    </button>

                    <Image width="16" height="16" layout="responsive"
                      src="/img/introduction-visual.png"
                      alt="Welcome"
                    />
                  </div>
                  <div className="intro--options">
                    <a className="ga">
                      <h3 className="gh">Simple User Interface</h3>
                      <p className="gp">
                        Our platform is designed to be incredibly user-friendly
                        and Easy to use.
                      </p>
                    </a>
                    <a className="ga">
                      <h3 className="gh">Reliable RoadMaps</h3>
                      <p className="gp">
                        Count on our crafted roadmaps that provides with
                        dependable guidance.
                      </p>
                    </a>
                    <a className="ga">
                      <h3 className="gh">Empowerment &amp; growth</h3>
                      <p className="gp">
                        Experience personal and professional growth like never
                        before.
                      </p>
                    </a>
                  </div>
                </div>
              </li>

              <li className="l-section section">
                <div className="about">
                  <div className="about--banner">
                    <h2>
                      We
                      <br />
                      believe in
                      <br />
                      Personal
                      <br />
                      Growth
                    </h2>
                    <a className="dvd" href="vid.html">
                      DEMO VIDEO
                      <span>
                        <svg
                          version="1.1"
                          id="Layer_1"
                          x="0px"
                          y="0px"
                          viewBox="0 0 150 118"
                        >
                          <g transform="translate(0.000000,118.000000) scale(0.100000,-0.100000)">
                            <path d="M870,1167c-34-17-55-57-46-90c3-15,81-100,194-211l187-185l-565-1c-431,0-571-3-590-13c-55-28-64-94-18-137c21-20,33-20,597-20h575l-192-193C800,103,794,94,849,39c20-20,39-29,61-29c28,0,63,30,298,262c147,144,272,271,279,282c30,51,23,60-219,304C947,1180,926,1196,870,1167z" />
                          </g>
                        </svg>
                      </span>
                    </a>
                    <br />
                    <br />
                    <a className="minf" href="pricing.html">
                      Pricing
                      <span></span>
                    </a>
                    <a className="minf" href="info.html">
                      More Info ...
                      <span></span>
                    </a>
                    <Image width="16" height="16" layout="responsive" src="/img/about-visual.png" alt="About Us" />
                  </div>

                  <div className="about--options">
                    <a href="info.html">
                      <h3>Initial</h3>
                    </a>
                    <a href="info.html">
                      <h3>Growth</h3>
                    </a>
                    <a href="info.html">
                      <h3>Success</h3>
                    </a>
                  </div>
                </div>
              </li>

              <li className="l-section section">
                <div className="wrapper">
                  <div>
                    <span className="dot"></span>
                  </div>
                  <div>
                    <span className="dot"></span>
                  </div>
                  <div>
                    <span className="dot"></span>
                  </div>
                  <div>
                    <span className="dot"></span>
                  </div>
                  <div>
                    <span className="dot"></span>
                  </div>
                  <div>
                    <span className="dot"></span>
                  </div>
                  <div>
                    <span className="dot"></span>
                  </div>
                  <div>
                    <span className="dot"></span>
                  </div>
                  <div>
                    <span className="dot"></span>
                  </div>
                  <div>
                    <span className="dot"></span>
                  </div>
                  <div>
                    <span className="dot"></span>
                  </div>
                  <div>
                    <span className="dot"></span>
                  </div>
                  <div>
                    <span className="dot"></span>
                  </div>
                  <div>
                    <span className="dot"></span>
                  </div>
                  <div>
                    <span className="dot"></span>
                  </div>
                </div>

                <div className="worky">
                  <div className="work">
                    <br />
                    <br />
                    <br />
                    <h2>
                      Try our
                      <span
                        style={{
                          color: "rgb(127, 0, 245)",
                          textShadow: "0 0px 50px rgba(54, 0, 90, 0.767)",
                        }}
                      >
                        Services
                      </span>
                    </h2>
                    <h2>:</h2>
                    <div className="try" id="try">
                      <a
                        href="rdm.html"
                        className="container"
                        id="TAWJIH"
                        style={{
                          textAlign: "center",
                          textDecoration: "none",
                          color: "black",
                        }}
                      >
                        <h1>
                          TAWJIHI<span>AI</span>🧭
                        </h1>
                      </a>
                      <a
                        href="form.html"
                        className="container"
                        id="ROAD"
                        style={{
                          textAlign: "center",
                          textDecoration: "none",
                          color: "black",
                        }}
                      >
                        <h1>
                          ROAD<span>MAPS</span>🚀
                        </h1>
                      </a>
                    </div>
                  </div>
                </div>
              </li>

              <li className="l-section section">
                <div className="contact">
                  <div className="contact--lockup">
                    <div className="modal">
                      <div className="modal--information">
                        <a href="https://mounseflit.surge.sh">
                          <Image width="16" height="16" layout="responsive" alt="1"
                            src="/img/logoml.png"
                            className="social-logo"
                          />
                        </a>
                        <p>
                          <a href="https://mounseflit.surge.sh">
                            THE M.A.C TEAM
                          </a>
                        </p>
                        <p>
                          <a href="https://www.google.com/maps/search/?api=1&amp;query=33.577321802376964,-7.552290636260293">
                            Morocco, Casablanca, 20700
                          </a>
                        </p>
                        <a href="mailto:myway.core.team@gmail.com">
                          myway.core.team@gmail.com
                        </a>
                        <a href="tel:+212614335196">+212614335196</a>
                      </div>
                      <ul className="modal--options">
                        <style>
                          {`
                                              .grid-layout {
                                                display: grid;
                                                grid-template-columns: repeat(4, 1fr);
                                                gap: 10px;
                                              }

                                              .grid-item {
                                                display: block;
                                                text-align: center;
                                                border-radius: 200px;
                                                transition: background-color 0.3s, color 0.3s;
                                              }

                                              .grid-item:hover {
                                                background-color: rgb(127, 0, 245);
                                                color: white;
                                              }

                                              .social-logo {
                                                width: 50px;
                                                height: 50px;
                                              }
                                            `}
                        </style>

                        <br />

                        <li>
                          <a>Our Social :</a>
                        </li>
                        <br />
                        <div className="grid-layout">
                          <a
                            href="https://linkedin.com/company/my-way-ma"
                            className="grid-item"
                          >
                            <Image width="16" height="16" layout="responsive"
                              src="https://d3nvy39jvu7woe.cloudfront.net/static/images/vcard_plus/social_link_icons/linkedin_icon.png"
                              className="social-logo" alt="1"
                            />
                          </a>

                          <a
                            href="http://twitter.com/mounseflit"
                            className="grid-item"
                          >
                            <Image width="16" height="16" layout="responsive"
                              src="https://d3nvy39jvu7woe.cloudfront.net/static/images/vcard_plus/social_link_icons/twitter_icon.png"
                              className="social-logo" alt="1"
                            />
                          </a>

                          <a
                            href="https://www.instagram.com/myway_ai/"
                            className="grid-item"
                          >
                            <Image width="16" height="16" layout="responsive"
                              src="https://d3nvy39jvu7woe.cloudfront.net/static/images/vcard_plus/social_link_icons/instagram_icon.png"
                              className="social-logo" alt="1"
                            />
                          </a>

                          <a
                            href="https://github.com/mounseflit"
                            className="grid-item"
                          >
                            <Image width="16" height="16" layout="responsive"
                              src="https://d3nvy39jvu7woe.cloudfront.net/static/images/vcard_plus/social_link_icons/github_icon.png"
                              className="social-logo" alt="1"
                            />
                          </a>
                        </div>

                        <br />
                      </ul>
                    </div>
                  </div>
                </div>
              </li>

              <li className="l-section section">
                <div className="hire">
                  <form className="work-request">
                    <div className="work-request--options">
                      <span className="options-a">
                        <input id="opt-1" type="checkbox" value="INVESTMENT" />
                        <label htmlFor="opt-1">
                          <svg
                            version="1.1"
                            id="Layer_1"
                            x="0px"
                            y="0px"
                            viewBox="0 0 150 111"
                          >
                            <g transform="translate(0.000000,111.000000) scale(0.100000,-0.100000)">
                              <path d="M950,705L555,310L360,505C253,612,160,700,155,700c-6,0-44-34-85-75l-75-75l278-278L550-5l475,475c261,261,475,480,475,485c0,13-132,145-145,145C1349,1100,1167,922,950,705z" />
                            </g>
                          </svg>
                          INVESTMENT
                        </label>

                        <input id="opt-2" type="checkbox" value="CAREERS" />
                        <label htmlFor="opt-2">
                          <svg
                            version="1.1"
                            id="Layer_1"
                            
                            
                            x="0px"
                            y="0px"
                            viewBox="0 0 150 111"
                            
                          >
                            <g transform="translate(0.000000,111.000000) scale(0.100000,-0.100000)">
                              <path d="M950,705L555,310L360,505C253,612,160,700,155,700c-6,0-44-34-85-75l-75-75l278-278L550-5l475,475c261,261,475,480,475,485c0,13-132,145-145,145C1349,1100,1167,922,950,705z" />
                            </g>
                          </svg>
                          CAREERS
                        </label>

                        <input id="opt-3" type="checkbox" value="SPONSORSHIP" />
                        <label htmlFor="opt-3">
                          <svg
                            version="1.1"
                            id="Layer_1"
                            
                            
                            x="0px"
                            y="0px"
                            viewBox="0 0 150 111"
                 
                            
                          >
                            <g transform="translate(0.000000,111.000000) scale(0.100000,-0.100000)">
                              <path d="M950,705L555,310L360,505C253,612,160,700,155,700c-6,0-44-34-85-75l-75-75l278-278L550-5l475,475c261,261,475,480,475,485c0,13-132,145-145,145C1349,1100,1167,922,950,705z" />
                            </g>
                          </svg>
                          SPONSORSHIP
                        </label>
                      </span>
                      <span className="options-b">
                        <input id="opt-4" type="checkbox" value=" CONTRIBUTE" />
                        <label htmlFor="opt-4">
                          <svg
                            version="1.1"
                            id="Layer_1"
                            
                            
                            x="0px"
                            y="0px"
                            viewBox="0 0 150 111"
                  
                            
                          >
                            <g transform="translate(0.000000,111.000000) scale(0.100000,-0.100000)">
                              <path d="M950,705L555,310L360,505C253,612,160,700,155,700c-6,0-44-34-85-75l-75-75l278-278L550-5l475,475c261,261,475,480,475,485c0,13-132,145-145,145C1349,1100,1167,922,950,705z" />
                            </g>
                          </svg>
                          CONTRIBUTE
                        </label>
                        <input id="opt-5" type="checkbox" value="API ACCESS" />
                        <label htmlFor="opt-5">
                          <svg
                            version="1.1"
                            id="Layer_1"
                            
                            
                            x="0px"
                            y="0px"
                            viewBox="0 0 150 111"
                   
                            
                          >
                            <g transform="translate(0.000000,111.000000) scale(0.100000,-0.100000)">
                              <path d="M950,705L555,310L360,505C253,612,160,700,155,700c-6,0-44-34-85-75l-75-75l278-278L550-5l475,475c261,261,475,480,475,485c0,13-132,145-145,145C1349,1100,1167,922,950,705z" />
                            </g>
                          </svg>
                          API ACCESS
                        </label>
                        <input id="opt-6" type="checkbox" value="HELP" />
                        <label htmlFor="opt-6">
                          <svg
                            version="1.1"
                            id="Layer_1"
                            
                            
                            x="0px"
                            y="0px"
                            viewBox="0 0 150 111"
    
                            
                          >
                            <g transform="translate(0.000000,111.000000) scale(0.100000,-0.100000)">
                              <path d="M950,705L555,310L360,505C253,612,160,700,155,700c-6,0-44-34-85-75l-75-75l278-278L550-5l475,475c261,261,475,480,475,485c0,13-132,145-145,145C1349,1100,1167,922,950,705z" />
                            </g>
                          </svg>
                          HELP
                        </label>
                      </span>
                    </div>
                    <div className="work-request--information">
                      <div className="information-name">
                        <input id="name" type="text" spellCheck={false} />
                        <label htmlFor="name">Name</label>
                      </div>
                      <div className="information-email">
                        <input id="email" type="email" spellCheck={false} />
                        <label htmlFor="email">Email</label>
                      </div>
                    </div>
                    <input id="btnsub" type="submit" value="Send Request" />
                  </form>
                </div>
              </li>
            </ul>
          </div>
        </div>
      </div>

      <ul className="outer-nav">
        <li className="is-active">Home</li>
        <li>About</li>
        <li>Try It</li>
        <li>Us</li>
        <li>Contact</li>
      </ul>
    </div>
  );

  

  return <>
        {mobilePage}
        {mainPage}

        <script src="https://ajax.googleapis.com/ajax/libs/jquery/2.2.4/jquery.min.js"></script>
        <script src="assets/js/functions-min.js"></script>
        <script src="https://smtpjs.com/v3/smtp.js"></script>
  </>;
}
